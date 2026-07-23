import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

with open("data/goldenset.json", encoding="utf-8") as f:
    golden = json.load(f)

chunk_article_ids = chunks["chunk_id"].str.split("_").str[0].astype(int).values
CANDIDATES = 25

def retrieve_and_rerank(query, k):
    q = embed_model.encode([query])[0]
    scores = embeddings @ q
    cand_idx = np.argsort(scores)[::-1][:CANDIDATES]

    pairs = [[query, chunks.iloc[i]["text"]] for i in cand_idx]
    rerank_scores = reranker.predict(pairs)
    reranked = cand_idx[np.argsort(rerank_scores)[::-1]]
    return chunk_article_ids[reranked[:k]]

for k in [1, 3, 5, 10]:
    hits = sum(
        item["answer_article_id"] in retrieve_and_rerank(item["question"], k)
        for item in golden
    )
    print(f"reranked recall@{k}: {hits/len(golden):.3f}  ({hits}/{len(golden)})")