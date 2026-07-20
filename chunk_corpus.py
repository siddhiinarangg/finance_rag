import pandas as pd

df = pd.read_parquet("corpus_clean.parquet")

CHUNK_SIZE = 800
OVERLAP = 100

def split_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

rows = []
for _, article in df.iterrows():
    pieces = split_text(article["Text"], CHUNK_SIZE, OVERLAP)
    for i, piece in enumerate(pieces):
        rows.append({
            "chunk_id": f"{article['id_']}_{i}",
            "text": piece,
            "symbol": article["symbol"],
            "company": article["company"],
            "title": article["Title"],
            "date": article["Publishdate"],
            "url": article["links"],
        })

chunks_df = pd.DataFrame(rows)
chunks_df.to_parquet("chunks.parquet")

print("Articles in:", len(df))
print("Chunks out:", len(chunks_df))
print("Average chunks per article:", round(len(chunks_df) / len(df), 1))
print("\nExample chunk:")
print(chunks_df.iloc[0]["text"][:300])
print("\nIts metadata:")
print(chunks_df.iloc[0][["chunk_id", "symbol", "company", "date"]])