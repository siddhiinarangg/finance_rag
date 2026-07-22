import os, json, time
import numpy as np, pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GEN_MODEL = "llama-3.1-8b-instant"

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)

with open("goldenset.json", encoding="utf-8") as f:
    answerable = [{"question": g["question"], "answerable": True} for g in json.load(f)]
with open("refusal_set.json", encoding="utf-8") as f:
    unanswerable = [{"question": r["question"], "answerable": False} for r in json.load(f)]

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
        context += f"[Source {i}] {r['company']} ({r['symbol']}): {r['text'][:400]}\n\n"
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
            print(f"  retry {attempt+1}: {str(e)[:60]}")
            time.sleep(10)
    return ""

def is_refusal(answer):
    a = answer.lower()
    markers = ["don't have information", "do not have information",
               "not in my sources", "sources do not contain",
               "sources don't contain", "no information"]
    return any(m in a for m in markers)

counts = {"correct_answer": 0, "hallucination": 0, "over_refusal": 0, "correct_refusal": 0}

for item in mixed:
    q, should_answer = item["question"], item["answerable"]
    answer = generate(q, retrieve(q))
    if not answer:
        continue
    refused = is_refusal(answer)

    if should_answer and not refused:
        counts["correct_answer"] += 1
    elif should_answer and refused:
        counts["over_refusal"] += 1
        print(f"[OVER-REFUSAL] {q}")
    elif not should_answer and not refused:
        counts["hallucination"] += 1
        print(f"[HALLUCINATION] {q}\n    -> {answer[:150]}")
    else:
        counts["correct_refusal"] += 1

    time.sleep(2)

n_ans = counts["correct_answer"] + counts["over_refusal"]
n_unans = counts["correct_refusal"] + counts["hallucination"]

print("\n--- confusion matrix ---")
print(f"correct answers:   {counts['correct_answer']}/{n_ans}")
print(f"over-refusals:     {counts['over_refusal']}/{n_ans}")
print(f"correct refusals:  {counts['correct_refusal']}/{n_unans}")
print(f"hallucinations:    {counts['hallucination']}/{n_unans}")
print(f"\noverall accuracy:  {(counts['correct_answer'] + counts['correct_refusal'])/(n_ans + n_unans):.3f}")