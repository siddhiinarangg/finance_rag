import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, top_k=5):
    query_vec = model.encode([query])[0]
    scores = embeddings @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    for rank, i in enumerate(top_idx, 1):
        row = chunks.iloc[i]
        print(f"\n--- Result {rank} | score {scores[i]:.3f} | {row['company']} ({row['symbol']}) ---")
        print(row["text"][:300])

while True:
    query = input("\nAsk a question (or type 'quit'): ")
    if query.lower() == "quit":
        break
    search(query)