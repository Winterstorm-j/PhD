#! .venv/bin/python3

import os
# BERTopic uses tokenisation so throws warnings if multiprocessing after
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
import util_functions as uf
from nltk.corpus import stopwords
from bertopic import BERTopic
from sklearn.metrics import silhouette_score


combined = pd.read_csv("modelReadyData.csv", encoding='utf-8').map(str).map(str.strip).reset_index(drop=True)
dataAsList = combined['allData'].to_list()

param_grid = [
    {"min_cluster_size": i, "min_samples": j, "cluster_selection_epsilon": k} 
    for i in range(5,26,5) 
    for j in [1,5,7]
    for k in uf.float_range(0,0.3,0.1)]

results = []
best_model = None
best_results = []
best_score = -np.inf


# Evaluate each configuration
for params in param_grid:
    print(f"\nTraining with params: {params}")

    # Create custom HDBSCAN model
    hdbscan_model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        cluster_selection_epsilon=params["cluster_selection_epsilon"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # Bertopic model instantiation
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        embedding_model = model,
        calculate_probabilities=True,
        nr_topics='auto',
        #seed_topic_list=topic_list
        )

    topics,probs = topic_model.fit_transform(dataAsList, embeddings=embeddings)
    new_topics = topic_model.reduce_outliers(dataAsList,
                                        topic_model.topics_, # type: ignore
                                        strategy="probabilities",
                                        probabilities=topic_model.probabilities_ ) # type: ignore

    # Evaluate model
    # Topic coherence
    coherence = uf.calculate_coherence_score(topic_model, dataAsList)

    # Topic diversity
    topic_words = topic_model.get_topics()
    diversity = uf.calculate_diversity_score(topic_model)

    # Silhouette score (only on clustered docs)
    valid_idx = [i for i, t in enumerate(topics) if t != -1]
    if len(valid_idx) > 2:
        sil_score = silhouette_score(
            np.array(embeddings)[valid_idx],
            np.array(topics)[valid_idx]
        )
    else:
        sil_score = -1

    score = coherence * 0.7 + diversity * 0.15 + sil_score * 0.15  # weighted scoring

    print(f"Coherence={coherence:.4f}, Diversity={diversity:.4f}, Silhouette={sil_score:.4f}, Score={score:.4f}")

    results.append((params, coherence, diversity, sil_score, score))

    # Keep best model
    if score > best_score:
        best_results = [params, new_topics, probs, coherence, diversity, sil_score, score]
        best_score= score
        best_model = topic_model
        

# Report & use best model
print("\nBest configuration:", best_model.hdbscan_model.get_params()) # type: ignore
print("Best score:", best_score)

# Save model
best_model.save("bertopic_optimized_hdbscan_5101",# type: ignore
                serialization="safetensors",
                save_embedding_model=True,
                save_ctfidf=True) 
import json

# A function to handle NumPy types during JSON serialization
def numpy_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # If the object is not a type we handle, raise a TypeError
    # The default encoder will then handle standard types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")



with open("best_model5101_params.json", "w") as f:
    json.dump(best_results, f, default=numpy_encoder )