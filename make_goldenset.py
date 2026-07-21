import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
TARGET = 40          # how many clean questions we want
POOL = 100           # over-sample this many articles (some get skipped)
MAX_ATTEMPTS = 3     # retries per article before giving up


def call(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def generate(company, symbol, text, feedback=None):
    correction = ""
    if feedback:
        correction = f"\nYour previous attempt was rejected because: {feedback}\nFix that and try again."

    prompt = f"""You are creating an evaluation question for a retrieval system.
Write ONE specific, factual question whose answer is a fact ABOUT {company} ({symbol}) stated in the article.

Rules:
- The main subject MUST be {company} itself, not another company merely mentioned.
- Name {company} explicitly. Never say "the company" or "this article".
- It must be answerable ONLY from this article, not from general knowledge.
- If the article is not actually about {company}, respond with exactly: SKIP
{correction}
Article: {text[:2000]}

Return only the question, or SKIP."""
    return call(prompt)


def validate(question, company, symbol, text):
    prompt = f"""Judge whether this question is a good retrieval-evaluation question for {company} ({symbol}).

Question: {question}

Article: {text[:2000]}

A good question meets ALL of:
1. Its main subject is {company}, not another company mentioned in passing.
2. It names {company} explicitly and stands alone (no "the company"/"this article").
3. Its answer comes from THIS article, not general world knowledge.

Reply in exactly this format:
VERDICT: PASS or FAIL
REASON: one short sentence"""
    out = call(prompt)
    verdict = "PASS" if "PASS" in out.split("REASON")[0].upper() else "FAIL"
    reason = out.split("REASON:")[-1].strip() if "REASON:" in out else ""
    return verdict, reason


articles = pd.read_parquet("corpus_clean.parquet")
pool = articles.sample(POOL, random_state=42).reset_index(drop=True)

golden = []
for _, row in pool.iterrows():
    if len(golden) >= TARGET:
        break

    company, symbol, text, aid = row["company"], row["symbol"], row["Text"], row["id_"]
    feedback = None
    kept = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        question = generate(company, symbol, text, feedback)

        if question.upper().startswith("SKIP") or len(question) < 15:
            print(f"[{symbol}] source rejected (not about company)")
            break

        verdict, reason = validate(question, company, symbol, text)
        if verdict == "PASS":
            golden.append({
                "question": question,
                "answer_article_id": int(aid),
                "company": company,
                "symbol": symbol,
            })
            print(f"[{symbol}] kept ({len(golden)}/{TARGET}): {question}")
            kept = True
            break
        else:
            print(f"[{symbol}] attempt {attempt} failed: {reason}")
            feedback = reason

        time.sleep(1)

    if not kept:
        time.sleep(1)

with open("goldenset.json", "w", encoding="utf-8") as f:
    json.dump(golden, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(golden)} validated questions to goldenset.json")