import os, json, time, re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
GEN_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "gemini-flash-latest"

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)

with open("goldenset.json", encoding="utf-8") as f:
    golden = json.load(f)[:15]

def retrieve(query, top_k=5, candidates=25):
    dense = embeddings @ embed_model.encode([query])[0]
    sparse = bm25.get_scores(query.lower().split())
    dn = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
    sn = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-9)
    cand = np.argsort(dn + sn)[::-1][:candidates]
    pairs = [[query, chunks.iloc[i]["text"]] for i in cand]
    order = cand[np.argsort(reranker.predict(pairs))[::-1]]
    return chunks.iloc[order[:top_k]]

def generate(query, hits):
    context = ""
    for i, (_, r) in enumerate(hits.iterrows(), 1):
        context += f"[Source {i}] {r['company']} ({r['symbol']}): {r['text'][:400]}\n\n"
    prompt = f"""Answer using ONLY the sources. Cite [Source N] after each claim.
If the answer isn't in the sources, say "I don't have information on that in my sources."

Sources:
{context}
Question: {query}
Answer:"""
    for attempt in range(3):
        try:
            r = groq_client.chat.completions.create(
                model=GEN_MODEL, messages=[{"role": "user", "content": prompt}]
            )
            return r.choices[0].message.content.strip(), context
        except Exception as e:
            print(f"  gen retry {attempt+1}: {str(e)[:80]}")
            time.sleep(10)
    return "", context

def judge(query, context, answer):
    prompt = f"""You are evaluating a RAG answer.

Question: {query}

Sources:
{context}
Answer: {answer}

Score each 1-5 (5 = best):
- FAITHFULNESS: are all claims supported by the sources (no made-up facts)?
- CITATION: do the [Source N] citations point to sources that actually support the claim?

Reply EXACTLY:
FAITHFULNESS: <1-5>
CITATION: <1-5>
REASON: <one sentence>"""
    for attempt in range(3):
        try:
            r = gemini.models.generate_content(model=JUDGE_MODEL, contents=prompt)
            return r.text.strip()
        except Exception as e:
            print(f"  judge retry {attempt+1}: {str(e)[:80]}")
            time.sleep(10)
    return "FAITHFULNESS: 0\nCITATION: 0\nREASON: judge unavailable"

def parse(text, label):
    m = re.search(rf"{label}:\s*([1-5])", text)
    return int(m.group(1)) if m else None

faith, cite = [], []
for item in golden:
    q = item["question"]
    hits = retrieve(q)
    answer, context = generate(q, hits)
    if not answer:
        print(f"SKIP (generation failed) | {q[:50]}")
        continue
    verdict = judge(q, context, answer)
    f, c = parse(verdict, "FAITHFULNESS"), parse(verdict, "CITATION")
    if f: faith.append(f)
    if c: cite.append(c)
    print(f"F={f} C={c} | {q[:55]}")
    time.sleep(3)

print(f"\nAvg faithfulness: {np.mean(faith):.2f} / 5   ({len(faith)} scored)")
print(f"Avg citation:     {np.mean(cite):.2f} / 5   ({len(cite)} scored)")