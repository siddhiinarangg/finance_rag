import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=5):
    query_vec = embed_model.encode([query])[0]
    scores = embeddings @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return chunks.iloc[top_idx]

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
        model="gemini-3-flash-preview",
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