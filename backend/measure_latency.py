import os
import json
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

t0 = time.time()
chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)
startup = time.time() - t0
print(f"startup (models + BM25 index): {startup:.2f}s\n")

with open("data/goldenset.json", encoding="utf-8") as f:
    questions = [g["question"] for g in json.load(f)][:10]

timings = {"embed": [], "dense": [], "bm25": [], "rerank": [], "llm": [], "total": []}

for q in questions:
    t_start = time.time()

    t = time.time()
    q_vec = embed_model.encode([q])[0]
    timings["embed"].append(time.time() - t)

    t = time.time()
    dense = embeddings @ q_vec
    timings["dense"].append(time.time() - t)

    t = time.time()
    sparse = bm25.get_scores(q.lower().split())
    timings["bm25"].append(time.time() - t)

    dn = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
    sn = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-9)
    cand = np.argsort(dn + sn)[::-1][:25]

    t = time.time()
    pairs = [[q, chunks.iloc[i]["text"]] for i in cand]
    order = cand[np.argsort(reranker.predict(pairs))[::-1]]
    timings["rerank"].append(time.time() - t)

    hits = chunks.iloc[order[:5]]
    context = ""
    for i, (_, r) in enumerate(hits.iterrows(), 1):
        context += f"[Source {i}] {r['company']} ({r['symbol']}): {r['text']}\n\n"

    prompt = f"""Answer using ONLY the sources. Cite [Source N] after each claim.
If the answer isn't in the sources, say "I don't have information on that in my sources."

Sources:
{context}
Question: {q}
Answer:"""

    t = time.time()
    try:
        client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
        timings["llm"].append(time.time() - t)
    except Exception as e:
        print(f"  llm call failed: {str(e)[:60]}")
        timings["llm"].append(float("nan"))

    timings["total"].append(time.time() - t_start)
    print(f"{timings['total'][-1]:6.2f}s  |  {q[:55]}")
    time.sleep(3)

print("\n--- per-query latency (mean over 10 queries) ---")
for stage in ["embed", "dense", "bm25", "rerank", "llm", "total"]:
    vals = [v for v in timings[stage] if not np.isnan(v)]
    if vals:
        pct = 100 * np.mean(vals) / np.mean(timings["total"])
        print(f"{stage:8s}: {np.mean(vals):6.3f}s   ({pct:4.1f}% of total)")

total_vals = timings["total"]
print(f"\nmin / median / max total: {min(total_vals):.2f}s / {np.median(total_vals):.2f}s / {max(total_vals):.2f}s")