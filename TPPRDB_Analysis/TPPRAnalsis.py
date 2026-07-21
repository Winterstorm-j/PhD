#! .venv/bin/python3
 
import os
import re
# os.chdir('./PhD/TPPRDB_Analysis')
# BERTopic uses tokenisation so throws warnings if multiprocessing after
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
import util_functions as uf
from nltk.corpus import stopwords
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import json
import umap
from bertopic.representation import ZeroShotClassification


combined = pd.read_csv("data/mergedDataDec.csv", encoding='utf-8').map(str).map(str.strip).reset_index(drop=True)

# combine title, keywords, abstract, relevance and trace type columns if they are not nan or empty into a single column
list(combined.columns)
cols = ['Title','Trace_Type', 'Study_Type', 'Keywords', 'Abstract', 'Exp_Conditions_and_Results',
       'Relevance_to_Canada', 'Addressed question', 'Activity context', 'Category', 'Specifications',
       'Variables of interest', 'stringency of control', 'No of individuals', 'Replicates per Individual and condition', 
       'Nucleic Acid', 'Bodily origin', 'depositor characteristics','Criteria for shedder status', 'Previous activities', 
       'Contact scenario', 'Primary substrate type', 'Primary substrate Material', 'Deposit', 'Delay (conditions)', 
       'Secondary substrate type', 'Secondary Substrate material', 'Type of secondary contact', 'Further transfer', 
       'Background DNA on sampled surface', 'Sampling time', 'Persistence (conditions)', 'Sampling method', 'Sampling area',
       'Extraction', 'DNA Quantification', 'Input for Profiling', 'Profiling', 'Reference samples', 
       'Profile interpretation and mixture analysis','RNA data interpretation', 'DNA Quantity', 'Profile Quality', 
       'Parameter used for comparison', 'Summary of results','Raised questions (by authors)', 'Cautionary remarks']

combined['allData'] = combined[cols].apply(uf._join_non_na, axis=1)

# The sentences to encode
dataAsList = combined['allData'].to_list()

combined.to_csv("modelReadyData.csv")

#set domain specific stop words (ie forensic, bayesian etc) as too general. At the trace type level we dont care about methods,
# within trace type we can look at these
forensic_stopwords = json.load(open("forensic_stopwords.json"))["stopwords"]

# Get the list of other language stop words
multilingual_stop_words = stopwords.words()

custom_stopwords = list(text.ENGLISH_STOP_WORDS.union(
    forensic_stopwords,
    multilingual_stop_words)
                        )

#preprocessed_docs = [uf.preprocess(doc, custom_stopwords) for doc in dataAsList]

# create vectorise method
vectorizer_model = CountVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1,5),
    min_df=0.2,
    max_df=0.6
)


# Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", 
    # prompts={
    #     "classification": "Classify the provided text into topic or theme of forensic sample type or discipline based on the provided \
    #         text. Do not include law enforcement, justice, statistics or type of crime as themes. Label topic with trace type or originating location ",
    #     "retrieval": "Retrieve semantically similar text, input is in multiple languages: ",
    #     "clustering": "Identify the topic or theme of forensic sample type or discipline based on the provided \
    #         text. Do not include law enforcement, justice, statistics or type of crime as themes. Label topic with trace type or originating location"
    # }
    )


# Calculate embeddings by calling model.encode() saves time later for BERTopic to avoid doing this internally
embeddings = model.encode(dataAsList)
print(embeddings.shape)

'''
Topic Modeling with BERTopic: Minimum Viable Example
References:
[1] https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html
[2] https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html
[3] https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html
'''

# Fine-tune the topic representations
representation_model = KeyBERTInspired(
    top_n_words=20,
    nr_repr_docs=500,
    nr_samples=1000,
    nr_candidate_words=2000)

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

# macOS has a bug with matrix multiplication that causes runtime warnings of zero division. 
    # Create custom HDBSCAN model
hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    min_samples=1,
    cluster_selection_epsilon=0.2,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# Bertopic model instantiation
topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    # representation_model = representation_model,
    zeroshot_topic_list=candidate_topics,
    zeroshot_min_similarity=0.8,
    embedding_model = model,
   #hdbscan_model=hdbscan_model,
    #calculate_probabilities=True,
    nr_topics=32,
    #seed_topic_list=topic_list
    )

# load optimised model
best_model=BERTopic.load("ZSMLbaseVec_230626", embedding_model=model)
topics,probs = best_model.transform(dataAsList)#, embeddings=embeddings)

# topic_model = best_model
topics,probs = topic_model.fit_transform(dataAsList)#, embeddings=embeddings)
new_topics = topic_model.reduce_outliers(dataAsList,
                                        topic_model.topics_, # type: ignore
                                        strategy="probabilities",
                                        probabilities=probs ) # type: ignore

topic_model.update_topics(dataAsList, topics=new_topics, vectorizer_model=vectorizer_model) # type: ignore


with open("best_model_params.json", "r") as f:
    best_results = json.load(f)

combined['T1_topic'] = topics #topic_model.topics_ best_results[1]
combined['T1_redux_topic'] = new_topics

#combined['T1_probs'] = best_results[2] # type: ignore

topic_info2 = topic_model.get_topic_info() # type: ignore
#topic_model.set_topic_labels(list(topic_info['Name']))

# Exclude the -1 topic (outliers) for labeling the main topics
topic_info = topic_info2[topic_info2['Topic'] != -1].reset_index(drop=True)
# topic_info['Topic'] = topic_info['Topic'].astype(int) + 1
# topic_info = topic_info[topic_info['Topic'] != -1].reset_index(drop=True)
topic_info = topic_info.reset_index(drop=True)

# associate colours with the topics
colours_list = [
'#f0e68c','#fa8072','#90ee90','#ff1493','#7b68ee','#d62728','#ffb6c1','#d8bfd8',
'#ff00ff','#1e90ff','#fa8072','#90ee90','#ff1493','#7b68ee','#f5f5dc','#ffb6c1',
'#daa520','#8fbc8f','#8b008b','#b03060','#ff0000','#ffff00','#7cfc00','#8a2be2',
'#00ff7f','#dc143c','#00ffff','#1e90ff','#da70d6','#b0c4de','#ff00ff','#1e90ff',
'#696969','#556b2f','#800000','#483d8b','#008000','#008b8b','#000080','#9acd32',
    # '#d8bfd8','#ff00ff','#1e90ff','#fa8072','#87ceeb','#ff1493','#7b68ee','#696969',
    #             '#98fb98','#7cfc00','#deb887','#40e0d0','#9400d3','#00ff7f','#dc143c','#0000ff',
    #             '#d2691e','#9acd32','#00008b','#7f007f','#b03060','#ff0000','#ffa500','#ffff00',
    #             '#da70d6','#2e8b57','#7f0000','#808000','#483d8b','#008000','#008080','#4682b4'
]
# [
    # '#d62728','#f3722c','#005073','#ffa15a','#ff97ff','#fecb52',
    # '#b6e880','#2ca02c','#4d908e','#43aa8b','#00cc96','#f5f511','#4cddc9',
    # '#90dbf4','#19d3f3','#718591','#1f77b4','#a3c4f3','#cfbaf0',
    # '#6a19b5','#636efa','#ab63fa','#b9fbc0','#ffcfd2'
#     ]


colour_dict=dict(zip(topic_info['Representation'].astype(str).tolist(),colours_list))

mapped_colours = topic_info['Representation'].astype(str).map(colour_dict).tolist() 

# intertopic distance map
dist_map = topic_model.visualize_topics(width =1200) # type: ignore

# make coords for the annotations. Add offset to x placement to avoid overlap
coords = np.column_stack((dist_map.data[0]['x'], dist_map.data[0]['y'])) # type: ignore

# add random offset to extracted coords
sizeref = dist_map.data[0]['marker']['sizeref'] # type: ignore
offset_x = coords[:,0] + (sizeref * np.random.normal(0,1.5,len(coords[:,0]))) # type: ignore
    # ( * sizeref * 2) + # type: ignore
    # (np.random.binomial(1,0.5) * sizeref * -2) ) # type: ignore
offset_y=coords[:,1] + (sizeref * np.random.normal(0,1.5,len(coords[:,1]))) # type: ignore
    # (np.random.binomial(1,0.5) * sizeref* 6) + # type: ignore
    # (np.random.binomial(1,0.5) * sizeref * -4) ) # type: ignore

positions = np.column_stack([offset_x, offset_y])


min_dist = 3 # minimum allowed distance between any label and any other object

positions = uf.resolve_overlaps(positions, coords, min_dist, repulsion_strength=0.5,
        marker_repulsion_strength=1, attraction_strength=0.08, max_step=2)

offset_x, offset_y = positions[:, 0], positions[:, 1]

# Add static labels as annotations to the Plotly figure
annotations = []
for index, row in topic_info.iterrows():
    idx=index
    annotations.append(
        dict(
            ax=offset_x[idx], # arrow tail pos # type: ignore
            ay=offset_y[idx], # type: ignore
            text=f"Topic {row['Topic']}", # Use topic name
            showarrow=True,
            x=coords[idx][0], #arrow head pos
            y=coords[idx][1],
            font=dict(size=10, color="black"),
            # Position the text slightly offset from the marker
            xanchor='left',
            yanchor='middle',
            xref="x",
            yref="y",
            arrowhead=1,
            axref= 'x',
            ayref='y'
            )
    )
#remove slider first so it only prints to console once
dist_map['layout'].pop('sliders')

#update markers with colours
dist_map.update_traces(
    marker=dict(color=mapped_colours),
    selector=dict(mode='markers'),
    # name=topic_info['Name'].tolist()
)

# update to add the static labels
dist_map.update_layout( 
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="bottom",
        x=1.02,
        xanchor="right",
        y=1
    ),
    title={
        'text': "Distance Map of Topics",
        'y':0.99,   
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    annotations=annotations
    )

# Display the figure
# dist_map.show(renderer="browser")
dist_map.write_html("test2.html", auto_open=True)
displayTable = topic_info[['Topic','Name','Count']]
# displayTable.to_html(index=False)
print(displayTable.to_latex(index=False))

# topic-terms barcharts
bar_fig = topic_model.visualize_barchart(top_n_topics=33, autoscale=True, width=350) # type: ignore
bar_fig.write_html("bar.html", auto_open=True)
hierarchical_topics = topic_model.hierarchical_topics(dataAsList) # type: ignore

# topics dendrogram
hierarch_fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics) # type: ignore
hierarch_fig.write_html("hierarchy.html", auto_open=True)

hier_doc_fig = topic_model.visualize_hierarchical_documents(dataAsList, hierarchical_topics, embeddings=embeddings)
hier_doc_fig.write_html("hier_docs.html", auto_open=True)

heatmap_fig = topic_model.visualize_heatmap()
heatmap_fig.write_html("heatmap.html", auto_open=True)

rank = topic_model.visualize_term_rank(custom_labels=topic_info['Topic'].tolist()) # type: ignore
rank.write_html("rank.html", auto_open=True)


# # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
# reduced_embeddings = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine').fit_transform(embeddings)
# reduced_topics = topic_model.visualize_documents(dataAsList, reduced_embeddings=np.array(reduced_embeddings))
# #.write_html("./PhD-Windows/TPPRDB_Analysis/reduced_projections.html")
# reduced_topics.show()

# time series analysis

# The data to encode
# datedData = combined[['Year','Title','Trace_Type','Keywords','Abstract','Exp_Conditions_and_Results','Relevance_to_Canada']]
# dataAsDatedList = datedData.apply(
#     lambda row: '; '.join(row.dropna().astype(str)), axis=1
# ).to_list()

# date = datedData.Year
# date[date=='s.d.'] = np.nan


# topics_over_time = topic_model.topics_over_time(dataAsDatedList, date.astype('str').to_list())
# model.visualize_topics_over_time(topics_over_time, topics=[range(1,21)])

# Save the model
# topic_model.save("models/TPPRDB_BERTopic_Model")    


# subset data to that assigned to topic 1 (trace DNA) and rerun process
TraceDataAsList = combined.filter(combined['T1_topic'] == 1)



sbert_model = best_model
sbert_umap_model = best_model
sbert_umap_Rep_model = best_model
MLbase_Vec_model = best_model
MLbaseVec_UMAP_model = best_model

coherence = uf.calculate_coherence_score(topic_model, dataAsList)

# embeddings = topic_model._extract_embeddings(dataAsList) # type: ignore
all_topic_words = []
topic_info4 = topic_model.get_topic_info()
valid_topics = topic_info4[topic_info4['Topic'] != -1]

for topic_id in valid_topics['Topic']: 
    words = [word for word, _ in topic_model.get_topic(topic_id)]
    all_topic_words.extend(words)
    
topic_corpus_string = " ".join(all_topic_words)

# 4. Calculate Topic Diversity using MTLD via lexicalrichness
if len(all_topic_words) > 10:  # Ensure we have enough tokens to compute MTLD
    lex = LexicalRichness(topic_corpus_string)
    diversity = lex.mtld()
else:
    diversity = 0.0 

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


