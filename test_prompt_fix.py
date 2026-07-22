import os
import json
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GEN_MODEL = "llama-3.1-8b-instant"

chunks = pd.read_parquet("chunks.parquet")
embeddings = np.load("embeddings.npy")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized = [t.lower().split() for t in chunks["text"].tolist()]
bm25 = BM25Okapi(tokenized)

with open("goldenset.json", encoding="utf-8") as f:
    golden = json.load(f)

failed = [
    "Bank of America Advantage Savings",
    "quarterly cash dividend on Costco",
    "Wells Fargo Active Cash",
    "Tapestry",
    "Constellation Brands",
    "Juniper Networks",
]

OLD_PROMPT = """Answer using ONLY the sources. Cite [Source N] after each claim.
If the answer isn't in the sources, say "I don't have information on that in my sources."

Sources:
{context}
Question: {query}
Answer:"""

NEW_PROMPT = """Answer using ONLY the sources below. Cite [Source N] after each claim.

Guidelines:
- If the information is stated in any source, answer it, even if stated only once.
- Source company labels are sometimes incorrect. If a label disagrees with the content, trust the content.
- Only say "I don't have information on that in my sources" if the answer genuinely does not appear in any source.

Sources:
{context}
Question: {query}
Answer:"""


def retrieve(query, top_k=5, candidates=25):
    dense = embeddings @ embed_model.encode([query])[0]
    sparse = bm25.get_scores(query.lower().split())
    dn = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
    sn = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-9)
    cand = np.argsort(dn + sn)[::-1][:candidates]
    pairs = [[query, chunks.iloc[i]["text"]] for i in cand]
    order = cand[np.argsort(reranker.predict(pairs))[::-1]]
    return chunks.iloc[order[:top_k]]


def build_context(hits):
    context = ""
    for i, (_, r) in enumerate(hits.iterrows(), 1):
        context += f"[Source {i}] {r['company']} ({r['symbol']}): {r['text'][:400]}\n\n"
    return context


def generate(prompt_template, query, context):
    prompt = prompt_template.format(context=context, query=query)
    for attempt in range(3):
        try:
            r = groq_client.chat.completions.create(
                model=GEN_MODEL, messages=[{"role": "user", "content": prompt}]
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    retry {attempt+1}: {str(e)[:60]}")
            time.sleep(10)
    return ""


def is_refusal(answer):
    a = answer.lower()
    markers = ["don't have information", "do not have information",
               "not in my sources", "sources do not contain",
               "sources don't contain", "no information"]
    return any(m in a for m in markers)


old_refusals = 0
new_refusals = 0
tested = 0

for item in golden:
    if not any(f in item["question"] for f in failed):
        continue

    q = item["question"]
    context = build_context(retrieve(q))
    tested += 1

    old_ans = generate(OLD_PROMPT, q, context)
    time.sleep(2)
    new_ans = generate(NEW_PROMPT, q, context)
    time.sleep(2)

    old_r = is_refusal(old_ans)
    new_r = is_refusal(new_ans)
    old_refusals += old_r
    new_refusals += new_r

    print("=" * 74)
    print(f"Q: {q[:70]}")
    print(f"  OLD: {'REFUSED' if old_r else 'ANSWERED'} -> {old_ans[:160]}")
    print(f"  NEW: {'REFUSED' if new_r else 'ANSWERED'} -> {new_ans[:160]}")
    print()

print("=" * 74)
print(f"old prompt refusals: {old_refusals}/{tested}")
print(f"new prompt refusals: {new_refusals}/{tested}")