import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

chunks = pd.read_parquet("chunks.parquet")
print("Chunks to embed:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    chunks["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
)

np.save("embeddings.npy", embeddings)
print("Embeddings shape:", embeddings.shape)