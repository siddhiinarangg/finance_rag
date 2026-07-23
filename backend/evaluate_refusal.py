import os, json, time
import numpy as np
import pandas as pd
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

with open("data/refusal_set.json", encoding="utf-8") as f:
    refusal_set = json.load(f)

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

results = {}
for item in refusal_set:
    q, t = item["question"], item["type"]
    answer = generate(q, retrieve(q))
    refused = is_refusal(answer)
    results.setdefault(t, []).append(refused)
    status = "REFUSED" if refused else "ANSWERED (leak)"
    print(f"[{status}] {q}")
    if not refused:
        print(f"    -> {answer[:150]}")
    time.sleep(2)

print("\n--- refusal rate by category ---")
total = []
for t, vals in results.items():
    total += vals
    print(f"{t:20s}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.2f}")
print(f"{'OVERALL':20s}: {sum(total)}/{len(total)} = {sum(total)/len(total):.2f}")