import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

with open("goldenset.json", encoding="utf-8") as f:
    golden = json.load(f)

chunk_article_ids = chunks["chunk_id"].str.split("_").str[0].astype(int).values

def top_articles(query, k=10, candidates=25):
    q = embed_model.encode([query])[0]
    scores = embeddings @ q
    cand_idx = np.argsort(scores)[::-1][:candidates]
    pairs = [[query, chunks.iloc[i]["text"]] for i in cand_idx]
    reranked = cand_idx[np.argsort(reranker.predict(pairs))[::-1]]
    return chunk_article_ids[reranked[:k]]

for item in golden:
    retrieved = top_articles(item["question"])
    if item["answer_article_id"] not in retrieved:
        print(f"\nMISS: [{item['symbol']}] {item['company']}")
        print(f"  Q: {item['question']}")
        print(f"  Want article: {item['answer_article_id']}")
        print(f"  Got articles: {list(retrieved)}")