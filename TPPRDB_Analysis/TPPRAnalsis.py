import pandas as pd
import numpy as np
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
embeddings = model.encode(dataAsList)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings) 
# print(similarities)

# create topic clusters
clusters = util.community_detection(embeddings, min_community_size=10, threshold=0.75)

'''
Topic Modeling with BERTopic: Minimum Viable Example
References:
[1] https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html
[2] https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html
[3] https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html
'''
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt



# Clustering model: See [2] for more details
cluster_model = HDBSCAN(min_cluster_size = 10, 
                        metric = 'euclidean', 
                        cluster_selection_method = 'eom', 
                        prediction_data = True)



# sns.displot(cluster_model.outlier_scores_[np.isfinite(cluster_model.outlier_scores_)], rug=True)

from bertopic.representation import KeyBERTInspired

# Fine-tune your topic representations
representation_model = KeyBERTInspired(
    top_n_words=10,
    nr_repr_docs=20,
    nr_samples=500,
    nr_candidate_words=100)

# BERTopic model
topic_model = BERTopic(representation_model=representation_model,#embedding_model = model,
                       hdbscan_model = cluster_model, seed_topic_list=topic_list)

# Fit the model on a corpus
topics, probs = topic_model.fit_transform(dataAsList)

topic_model.get_topic_info()

# Visualization examples: See [3] for more details

# Save intertopic distance map as HTML file
topic_model.visualize_topics()#.write_html("./PhD-Windows/TPPRDB_Analysis/intertopic_dist_map.html")

# Save topic-terms barcharts as HTML file
topic_model.visualize_barchart(top_n_topics = 30)#.write_html("./PhD-Windows/TPPRDB_Analysis/barchart.html")

# Save documents projection as HTML file
topic_model.visualize_documents(docs=dataAsList, topics=topics)#.write_html("./PhD-Windows/TPPRDB_Analysis/projections.html")

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(dataAsList, reduced_embeddings=reduced_embeddings)#.write_html("./PhD-Windows/TPPRDB_Analysis/reduced_projections.html")

hierarchical_topics = topic_model.hierarchical_topics(dataAsList)
# Save topics dendrogram as HTML file
topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)#.write_html("./PhD-Windows/TPPRDB_Analysis/hieararchy.html")

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