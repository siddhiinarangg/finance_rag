import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)

with open("data/goldenset.json", encoding="utf-8") as f:
    golden = json.load(f)

failed = [
    "Bank of America Advantage Savings",
    "quarterly cash dividend on Costco",
    "Wells Fargo Active Cash",
    "Tapestry",
    "Constellation Brands",
    "Juniper Networks",
]

chunk_article_ids = chunks["chunk_id"].str.split("_").str[0].astype(int).values


def retrieve(query, top_k=5, candidates=25):
    dense = embeddings @ embed_model.encode([query])[0]
    sparse = bm25.get_scores(query.lower().split())
    dn = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
    sn = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-9)
    cand = np.argsort(dn + sn)[::-1][:candidates]
    pairs = [[query, chunks.iloc[i]["text"]] for i in cand]
    order = cand[np.argsort(reranker.predict(pairs))[::-1]]
    return order[:top_k]


for item in golden:
    if not any(f in item["question"] for f in failed):
        continue

    q = item["question"]
    want = item["answer_article_id"]
    idx = retrieve(q)
    got_articles = chunk_article_ids[idx]

    print("=" * 78)
    print(f"Q: {q}")
    print(f"want article {want} | retrieved: {list(got_articles)}")
    print(f"correct article in top 5: {want in got_articles}")

    for rank, i in enumerate(idx, 1):
        row = chunks.iloc[i]
        marker = "  <-- TARGET ARTICLE" if chunk_article_ids[i] == want else ""
        print(f"\n  [{rank}] {row['company']} ({row['symbol']}) art={chunk_article_ids[i]}{marker}")
        print(f"      {row['text'][:300]}")
    print()

    if want not in got_articles:
        target_chunks = chunks[chunk_article_ids == want]
        if len(target_chunks):
            print(f"  --- what the TARGET article actually contains ---")
            print(f"      {target_chunks.iloc[0]['text'][:400]}")
        else:
            print(f"  --- article {want} not found in chunks ---")
        print()