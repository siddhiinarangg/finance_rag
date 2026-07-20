from datasets import load_dataset

df = load_dataset("KrossKinetic/SP500-Financial-News-Articles-Time-Series", split="train").to_pandas()

print("Rows:", len(df))
print("Columns:", list(df.columns))
print("Date range:", df["Publishdate"].min(), "to", df["Publishdate"].max())
print("Unique companies:", df["symbol"].nunique())

df["text_len"] = df["Text"].str.len()
print("\nText length min / median / max:", df["text_len"].min(), int(df["text_len"].median()), df["text_len"].max())
print("Articles under 200 chars:", (df["text_len"] < 200).sum())

print("\nArticles per company (top 20):")
print(df["symbol"].value_counts().head(20))