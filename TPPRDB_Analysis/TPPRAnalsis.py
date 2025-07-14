#! ~\AppData\Local\Programs\Python\Python312\Projects\python

import pandas as pd
import numpy as np
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import re

TPPR_db = pd.read_csv("./PhD-Windows/TPPRDB_Analysis/TTADB_Feb2025_cleaned.csv").map(str)

TPPR_db.head

''' Likely need to change the data into a dict so that the papers are objects with properties in order to put into the db'''

# combine title, keywords, abstract, relevance and trace type columns into a single column
TPPR_db.columns
TPPR_db['allData'] = TPPR_db[['Title','Trace_Type','Keywords','Abstract','Exp_Conditions_and_Results','Relevance_to_Canada']].agg('; '.join, axis=1)

# The sentences to encode
dataAsList = TPPR_db['allData'].astype('str').to_list()

#set domain specific stop words (ie forensic, bayesian etc)
custom_stopwords = ['forensic', 'bayesian', 'analysis','samples',
                    'evidence', 'examination', 'investigation','sample', 'examined'
                    'crime', 'method', 'testing', 'probabilistic', 'probabilitiy', 
                    'interpretation', 'likelihood', 'ratio','collected','analyze', 
                    'analyse', 'analysis', 'analyzed','extracted','specimens', 'spectrometry', 'extraction', 
                    'examiners','analyzing', 'findings', 'propositions', 'chromatography']

def preprocess(text):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return ' '.join([word for word in words if word not in custom_stopwords])

preprocessed_docs = [preprocess(doc) for doc in dataAsList]


vectorizer_model = CountVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9
)


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2", 
    prompts={
        "classification": "Classify the following text: ",
        "retrieval": "Retrieve semantically similar text: ",
        "clustering": "Identify the topic or theme based on the text: ",
    })


# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(dataAsList)
print(embeddings.shape)
# [3, 384]

# # 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings) 
# # print(similarities)

# # create topic clusters
# clusters = util.community_detection(embeddings, min_community_size=10, threshold=0.75)

'''
Topic Modeling with BERTopic: Minimum Viable Example
References:
[1] https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html
[2] https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html
[3] https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html
'''

# sns.displot(cluster_model.outlier_scores_[np.isfinite(cluster_model.outlier_scores_)], rug=True)

# Fine-tune your topic representations
representation_model = KeyBERTInspired(
    top_n_words=50,
    nr_repr_docs=100,
    nr_samples=1000,
    nr_candidate_words=1000)

# Clustering model: See [2] for more details
cluster_model = HDBSCAN(min_cluster_size = 2, 
                        metric = 'euclidean', 
                        cluster_selection_method = 'eom', 
                        prediction_data = True)

# BERTopic model
topic_list=[['paints', 'paint','pigments','pigment',
     'colour','coating','coatings'],
['glass', 'glasses','refractive','windshields'],
['arson','petrol','gasoline', 'fires','hydrocarbons',
     'fire', 'liquids', 'ignitable'],
['methamphetamine','methadone','cocaine','drug', 'drugs',
     'metabolites','amphetamine','heroin','metabolite'],
['lipstick','cosmetics','lipsticks','cosmetic','lip','glitter'],
['pollen','botany','vegetation','spores','palynological','palynology','plants'],
['fingerprints','fingerprint','fingermarks','fingermark','ngerprints'],
['biomarkers','rna','mrna','luminol','haemoglobin'],
['bloodstain','patterns','blood','pattern','arterial','bpa'],
['hairs','hair'],
['dna','contamination', 'contaminations'],
['remains','skeletal','bones','cadavers','teeth',
      'dental','tooth','dentine','dentistry', 'dentin'],
['sperm','spermatozoa','semen','seminal', 'postcoital','vaginal'],
['soils','soil','dust','mineralogy','mineralogical',
      'geology','mineral','geoscience'],
['dna','mixture','mixtures'],
['condom','condoms','lubricants','lubricant',
       'condomlubricants', 'lubricated'],
['tape','tapes','adhesive'],
['fingernails','nails','fingernail','nail','scratching'],
['saliva','salivary','mouth'],
['fibres','fabrics','textile','cotton','fibre','fibers', 
      'garments', 'fabric','nylon','wool','dyed','polyester','shirt',
      'garment','fiber','textile', 'cloth','clothing','clothes',
     'material'],
['cartridge','ammunition','ammunitions','bullet','cartridges'],
['laundry','washing'],
['dna','wearer','scalp','trace','touch','skin','hands','touched'],
['gsr','gunshot','gunpowder','elemental','primer','residue',
      'powder','residue','residues','compounds'],
['explosive','explosives','bombs','bomb','chemical'],
['firearm','handguns','rifles',
     'pistol','shotguns','gun','shotgun'],
['corde','rope']]

topic_model = BERTopic(vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    #embedding_model = model,
    hdbscan_model = cluster_model, 
    seed_topic_list=topic_list)

# Fit the model on a corpus
topics, probs = topic_model.fit_transform(dataAsList)

hierarchical_topics = topic_model.hierarchical_topics(dataAsList)
# Save topics dendrogram as HTML file
topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)#.write_html("./PhD-Windows/TPPRDB_Analysis/hieararchy.html")

topic_model.get_topic_info()

# Save intertopic distance map as HTML file
topic_model.visualize_topics()#.write_html("./PhD-Windows/TPPRDB_Analysis/intertopic_dist_map.html")

# Save topic-terms barcharts as HTML file
topic_model.visualize_barchart(top_n_topics = 30)#.write_html("./PhD-Windows/TPPRDB_Analysis/barchart.html")

# Save documents projection as HTML file
topic_model.visualize_documents(docs=dataAsList, topics=topics)#.write_html("./PhD-Windows/TPPRDB_Analysis/projections.html")

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(dataAsList, reduced_embeddings=reduced_embeddings)#.write_html("./PhD-Windows/TPPRDB_Analysis/reduced_projections.html")

plt.scatter(dataAsList, s=50, linewidth=0, c='b', alpha=0.25)



# time series analysis

# The data to encode
datedData = TPPR_db[['Year','Title','Trace_Type','Keywords','Abstract','Exp_Conditions_and_Results','Relevance_to_Canada']].filter(
    like=
)
dataAsDatedList = datedData.apply(
    lambda row: '; '.join(row.dropna().astype(str)), axis=1
).to_list()

date = datedData.Year
date[date=='s.d.'] = np.NaN


topics_over_time = topic_model.topics_over_time(dataAsDatedList, date)
model.visualize_topics_over_time(topics_over_time, topics=[range(1,21)])