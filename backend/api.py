import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv(dotenv_path="../.env")

MODEL = "gemini-3.1-flash-lite"

app = FastAPI(title="Financial News RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# loaded once at startup, reused for every request
print("loading models and index...")
_t = time.time()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)

print(f"ready in {time.time() - _t:.1f}s")


class Question(BaseModel):
    question: str


class Source(BaseModel):
    n: int
    company: str
    symbol: str
    date: str
    url: str
    excerpt: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    refused: bool
    elapsed: float


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


def is_refusal(text):
    t = text.lower()
    markers = ["don't have information", "do not have information",
               "not in my sources", "sources do not contain",
               "sources don't contain"]
    return any(m in t for m in markers)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    started = time.time()
    query = payload.question.strip()

    if not query:
        return Answer(answer="Please enter a question.", sources=[],
                      refused=True, elapsed=0.0)

    hits = retrieve(query)

    context = ""
    sources = []
    for i, (_, row) in enumerate(hits.iterrows(), 1):
        context += f"[Source {i}] {row['company']} ({row['symbol']}), {row['date']}:\n{row['text']}\n\n"
        sources.append(Source(
            n=i,
            company=str(row["company"]),
            symbol=str(row["symbol"]),
            date=str(row["date"])[:10],
            url=str(row["url"]),
            excerpt=str(row["text"])[:220] + "...",
        ))

    prompt = f"""You are a financial news assistant. Answer the question using ONLY the sources below.
Cite the source number in brackets like [Source 1] after each claim. Use a separate bracket for each source, for example [Source 1] [Source 2], never [Source 1, Source 2].
If the sources do not contain the answer, say "I don't have information on that in my sources."

Sources:
{context}

Question: {query}

Answer:"""

    text = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            text = response.text
            break
        except Exception:
            if attempt == 2:
                return Answer(
                    answer="The language model is unavailable right now. Please try again in a moment.",
                    sources=[], refused=True,
                    elapsed=round(time.time() - started, 2),
                )
            time.sleep(3)

    refused = is_refusal(text)

    return Answer(
        answer=text,
        sources=[] if refused else sources,
        refused=refused,
        elapsed=round(time.time() - started, 2),
    )