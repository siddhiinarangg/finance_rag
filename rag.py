import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)

def retrieve(query, top_k=5, candidates=25):
    q_vec = embed_model.encode([query])[0]
    dense = embeddings @ q_vec
    sparse = bm25.get_scores(query.lower().split())

    dense_n = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
    sparse_n = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-9)
    combined = dense_n + sparse_n

    cand_idx = np.argsort(combined)[::-1][:candidates]
    pairs = [[query, chunks.iloc[i]["text"]] for i in cand_idx]
    rerank_scores = reranker.predict(pairs)
    reranked = cand_idx[np.argsort(rerank_scores)[::-1]]
    return chunks.iloc[reranked[:top_k]]

def answer(query):
    hits = retrieve(query)

    context = ""
    for i, (_, row) in enumerate(hits.iterrows(), 1):
        context += f"[Source {i}] {row['company']} ({row['symbol']}), {row['date']}:\n{row['text']}\n\n"

    prompt = f"""You are a financial news assistant. Answer the question using ONLY the sources below.
Cite the source number in brackets like [Source 1] after each claim.
If the sources do not contain the answer, say "I don't have information on that in my sources."

Sources:
{context}

Question: {query}

Answer:"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    print("\n" + "="*60)
    print("ANSWER:\n")
    print(response.text)
    print("\n" + "-"*60)
    print("SOURCES USED:")
    for i, (_, row) in enumerate(hits.iterrows(), 1):
        print(f"[Source {i}] {row['company']} ({row['symbol']}) — {row['date']} — {row['url']}")

while True:
    query = input("\nAsk a question (or 'quit'): ")
    if query.lower() == "quit":
        break
    answer(query)