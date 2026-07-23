from datasets import load_dataset

df = load_dataset("KrossKinetic/SP500-Financial-News-Articles-Time-Series", split="train").to_pandas()
print("Raw rows:", len(df))

df = df.drop_duplicates()
df = df[df["Text"].str.len() >= 200]
df = df.reset_index(drop=True)

df.to_parquet("corpus_clean.parquet")
print("Clean rows saved:", len(df))
print("Shortest article now:", df["Text"].str.len().min())
print("Saved to corpus_clean.parquet")