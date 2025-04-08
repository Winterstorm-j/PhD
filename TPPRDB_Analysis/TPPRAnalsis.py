import pandas as pd
import polars as pl
import numpy as nm
import torch

TPPR_db = pd.read_csv("./PhD-Windows/TPPRDB_Analysis/TTADB_Feb2025_cleaned.csv")
TPPR_db.head

''' Likely need to change the data into a dict so that the papers are objects with properties in order to put into the db'''

from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
titleAsList = TPPR_db['Title'].to_list()
# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(titleAsList)
print(embeddings)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings['Title'], embeddings['Title']) 

print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])