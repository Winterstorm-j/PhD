import pandas as pd
import polars as pl
import numpy as nm
import torch

TPPR_db = pd.read_csv("./PhD-Windows/TPPRDB_Analysis/TTADB_Feb2025_cleaned.csv").map(str)

TPPR_db.head


''' Likely need to change the data into a dict so that the papers are objects with properties in order to put into the db'''

# combine title, keywords, abstract, relevance and trace type columns into a single column
TPPR_db.columns
TPPR_db['allData'] = TPPR_db[['Title','Trace_Type','Keywords','Abstract','Exp_Conditions_and_Results','Relevance_to_Canada']].agg('; '.join, axis=1)


from sentence_transformers import SentenceTransformer, util


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2", 
    prompts={
        "classification": "Classify the following text: ",
        "retrieval": "Retrieve semantically similar text: ",
        "clustering": "Identify the topic or theme based on the text: ",
    })

# The sentences to encode
dataAsList = TPPR_db['allData'].astype('str').to_list()

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(dataAsList, prompt_name='clustering')
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings) 
print(similarities)


clusters = util.community_detection(embeddings, min_community_size=25, threshold=0.75)

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
    for sentence_id in cluster:
        print("\t", dataAsList[sentence_id])
    print("\t", "...")
