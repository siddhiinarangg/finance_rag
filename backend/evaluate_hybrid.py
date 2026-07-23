import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

with open("data/goldenset.json", encoding="utf-8") as f:
    golden = json.load(f)

chunk_article_ids = chunks["chunk_id"].str.split("_").str[0].astype(int).values

tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)

def hybrid_candidates(query, candidates=25):
    q_vec = embed_model.encode([query])[0]
    dense = embeddings @ q_vec
    sparse = bm25.get_scores(query.lower().split())

    dense_n = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
    sparse_n = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-9)
    combined = dense_n + sparse_n

    return np.argsort(combined)[::-1][:candidates]

def retrieve(query, k, candidates=25):
    cand_idx = hybrid_candidates(query, candidates)
    pairs = [[query, chunks.iloc[i]["text"]] for i in cand_idx]
    rr = reranker.predict(pairs)
    reranked = cand_idx[np.argsort(rr)[::-1]]
    return chunk_article_ids[reranked[:k]]

for k in [1, 3, 5, 10]:
    hits = sum(
        item["answer_article_id"] in retrieve(item["question"], k)
        for item in golden
    )
    print(f"hybrid recall@{k}: {hits/len(golden):.3f}  ({hits}/{len(golden)})")