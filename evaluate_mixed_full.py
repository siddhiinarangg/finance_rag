import os
import json
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GEN_MODEL = "llama-3.1-8b-instant"

# balanced subset to stay under the free-tier token budget while using FULL chunks
N_ANSWERABLE = 20
N_UNANSWERABLE = 12

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)

with open("goldenset.json", encoding="utf-8") as f:
    answerable = [{"question": g["question"], "answerable": True}
                  for g in json.load(f)][:N_ANSWERABLE]

with open("refusal_set.json", encoding="utf-8") as f:
    refusal = json.load(f)

# take a spread across all three refusal categories
by_type = {}
for r in refusal:
    by_type.setdefault(r["type"], []).append(r)
unanswerable = []
per_type = max(1, N_UNANSWERABLE // len(by_type))
for t, items in by_type.items():
    unanswerable += [{"question": i["question"], "answerable": False} for i in items[:per_type]]

mixed = answerable + unanswerable


def retrieve(query, top_k=5, candidates=25):
    dense = embeddings @ embed_model.encode([query])[0]
    sparse = bm25.get_scores(query.lower().split())
    dn = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
    sn = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-9)
    cand = np.argsort(dn + sn)[::-1][:candidates]
    pairs = [[query, chunks.iloc[i]["text"]] for i in cand]
    order = cand[np.argsort(reranker.predict(pairs))[::-1]]
    return chunks.iloc[order[:top_k]]


def generate(query, hits):
    context = ""
    for i, (_, r) in enumerate(hits.iterrows(), 1):
        # FULL chunk text - no truncation, matches rag.py
        context += f"[Source {i}] {r['company']} ({r['symbol']}): {r['text']}\n\n"

    prompt = f"""Answer using ONLY the sources. Cite [Source N] after each claim.
If the answer isn't in the sources, say "I don't have information on that in my sources."

Sources:
{context}
Question: {query}
Answer:"""

    for attempt in range(3):
        try:
            r = groq_client.chat.completions.create(
                model=GEN_MODEL, messages=[{"role": "user", "content": prompt}]
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  retry {attempt+1}: {str(e)[:70]}")
            time.sleep(15)
    return ""


def is_refusal(answer):
    a = answer.lower()
    markers = ["don't have information", "do not have information",
               "not in my sources", "sources do not contain",
               "sources don't contain", "no information"]
    return any(m in a for m in markers)


counts = {"correct_answer": 0, "hallucination": 0,
          "over_refusal": 0, "correct_refusal": 0}
skipped = 0

for item in mixed:
    q, should_answer = item["question"], item["answerable"]
    answer = generate(q, retrieve(q))

    if not answer:
        skipped += 1
        print(f"[SKIPPED - api failure] {q[:60]}")
        continue

    refused = is_refusal(answer)

    if should_answer and not refused:
        counts["correct_answer"] += 1
    elif should_answer and refused:
        counts["over_refusal"] += 1
        print(f"[OVER-REFUSAL] {q[:70]}")
    elif not should_answer and not refused:
        counts["hallucination"] += 1
        print(f"[HALLUCINATION] {q[:70]}\n    -> {answer[:150]}")
    else:
        counts["correct_refusal"] += 1

    time.sleep(2)

n_ans = counts["correct_answer"] + counts["over_refusal"]
n_unans = counts["correct_refusal"] + counts["hallucination"]

print("\n--- confusion matrix (FULL chunks, no truncation) ---")
if n_ans:
    print(f"correct answers:   {counts['correct_answer']}/{n_ans} = {counts['correct_answer']/n_ans:.3f}")
    print(f"over-refusals:     {counts['over_refusal']}/{n_ans} = {counts['over_refusal']/n_ans:.3f}")
if n_unans:
    print(f"correct refusals:  {counts['correct_refusal']}/{n_unans} = {counts['correct_refusal']/n_unans:.3f}")
    print(f"hallucinations:    {counts['hallucination']}/{n_unans} = {counts['hallucination']/n_unans:.3f}")
if n_ans + n_unans:
    acc = (counts["correct_answer"] + counts["correct_refusal"]) / (n_ans + n_unans)
    print(f"\noverall accuracy:  {acc:.3f}")
if skipped:
    print(f"skipped (api):     {skipped}")

print("\ncompare to 400-char run: over-refusals were 6/33 = 0.182")