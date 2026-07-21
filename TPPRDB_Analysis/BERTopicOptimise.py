#! .venv/bin/python3

from multiprocessing import freeze_support, set_start_method
import os

#  BERTopic uses tokenisation so throws warnings if multiprocessing after
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
import util_functions as uf
from nltk.corpus import stopwords
from bertopic import BERTopic
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
import json

combined = pd.read_csv("data/modelReadyData.csv", encoding='utf-8').map(str).map(str.strip).reset_index(drop=True)
dataAsList = combined['allData'].to_list()


#set domain specific stop words (ie forensic, bayesian etc) as too general. At the trace type level we dont care about methods,
# within trace type we can look at these
forensic_stopwords = json.load(open("forensic_stopwords.json"))["stopwords"]

# Get the list of other language stop words
multilingual_stop_words = stopwords.words()

custom_stopwords = list(text.ENGLISH_STOP_WORDS.union(
    forensic_stopwords,
    multilingual_stop_words)
                        )

# create vectorise method
vectorizer_model = CountVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1,5),
    min_df=0.2,
    max_df=0.6
)

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Calculate embeddings by calling model.encode() saves time later for BERTopic to avoid doing this internally
embeddings = model.encode(dataAsList)

# guidance for topic definitions
candidate_topics = [
    'paint, pigment, colour, color, coating, coatings', 
    'glass, refractive material, glasses, refractive, windshields', 
    'drugs, metabolite, precursor, narcotic, controlled substance, illicit, pharmaceutical, drug, drugs, metabolites',  
    'cosmetics, skincare, lotions, lipstick, glitter', 
    'pollen, vegetation, botany, spores, palynological, palynology, plants', 
    'fingerprint, fingermark, latent, enhancement', 
    'rna, body fluid, biomarker', 
    'bloodstain, blood, pattern, arterial, bpa', 
    'hair, hairs', 
    'skin,trace dna, touch dna, wearer, scalp, trace, touch, skin, hands, touched', 
    'blood, haemoglobin',
    'remains, skeletal, bones, cadavers, teeth, dental, tooth, dentine, dentistry, dentin, tissue, bone', 
    'sperm, seminal, semen, postcoital, vaginal', 
    'soils, soil, mineralogy, dust, geology, geoscience', 
    'condom, lubricant', 
    'tape, adhesive, glue', 
    'fingernail, nail', 
    'saliva, mouth', 
    'fibre, fiber, fabric, textile, cotton, garments, nylon, wool, dyed, polyester, shirt, garment, cloth, clothing, clothes, material',
    'cartridge, ammunition, ammunitions, bullet, cartridges', 
    'laundry, washing', 
    'dna', 
    'protein, proteomic',
    'gsr, gunshot residue, gunpowder, elemental, residues, compounds', 
    'explosive device, bomb, explosives', 
    'arson, ignitable liquid, accelerant, petrol, gasoline, fires,hydrocarbons',
    'firearm, handgun, rifle, pistol, shotgun, gun, weapon',
    'cord, rope',
    'digital forensics, mobile phone, computer, device, file, data',
    'psychology, mental trauma, psychiatry',
    'postmortem,autopsy, injury, wound']

# The BERTopic training helper now manages model construction directly, so a separate HDBSCAN instance is not required.

# param_grid = [
#     {"min_cluster_size":5, "min_samples": 1, "cluster_selection_epsilon": 0.2} for _ in range(50)]
#    # for i in range(5,11,5) 
#    # for k in range(2,11,2)
#     #for e in uf.float_range(0.1,0.3,0.1)]
    

results = [] 
best_model = None
best_results = []
best_score = -np.inf
  

# # # Alternative: Print the entire configuration dictionary at once
# # list(best_model.get_params())

topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    umap_model=UMAP(random_state=38),
    zeroshot_topic_list=candidate_topics,
    zeroshot_min_similarity=0.8,
    embedding_model = model,
    nr_topics=29
)

topics, _ = topic_model.fit_transform(dataAsList, embeddings=embeddings)

# # Evaluate each configuration
for topic_num in range(24, 36):
    print(topic_num)
        
# for pars in param_grid:
#     print(f"\nTraining with params: {pars}")

    runmodel = uf.run_model(
        docs=dataAsList,
        embeddings=embeddings,
        vectorizer_model=vectorizer_model,
        notopics=topic_num,
        candidate_topics=candidate_topics,
        embedding_model=model,
    )

    # Keep best model
    if runmodel[0][0][4] > best_score:
        best_results = runmodel[0]
        best_score = runmodel[0][0][4]
        best_model = runmodel[1]


# # Report & use best model
# print("\nBest configuration:", best_model.get_params()) # type: ignore
# print("Best score:", best_score)

# # Save model
# best_model.save("ZSMLbaseVec_230626",# type: ignore
#                 serialization="safetensors",
#                 save_embedding_model=True,
#                 save_ctfidf=True) 
# import json

# # A function to handle NumPy types during JSON serialization
# def numpy_encoder(obj):
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     # If the object is not a type we handle, raise a TypeError
#     # The default encoder will then handle standard types
#     raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# with open("ZSMLbaseVec_230626_params.json", "w") as f:
#     json.dump(best_results, f, default=numpy_encoder )

# CV of bertopic output with best parameters
cv_results = uf.cross_validate_zeroshot_bertopic(
    docs=dataAsList, 
    embeddings = embeddings,
    vectorizer_model=vectorizer_model,
    embedding_model= model,
    topic_list = candidate_topics
    ) 


if __name__ == '__main__':
    freeze_support()
    set_start_method('forkserver')