import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("goldenset.json", encoding="utf-8") as f:
    golden = json.load(f)

chunk_article_ids = chunks["chunk_id"].str.split("_").str[0].astype(int).values

def retrieve_article_ids(query, k):
    q = model.encode([query])[0]
    scores = embeddings @ q
    top_idx = np.argsort(scores)[::-1][:k]
    return chunk_article_ids[top_idx]

for k in [1, 3, 5, 10]:
    hits = 0
    for item in golden:
        retrieved = retrieve_article_ids(item["question"], k)
        if item["answer_article_id"] in retrieved:
            hits += 1
    recall = hits / len(golden)
    print(f"recall@{k}: {recall:.3f}  ({hits}/{len(golden)})")