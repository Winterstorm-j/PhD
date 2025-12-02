#! .venv/bin/python3
 
import os
# os.chdir('./PhD/TPPRDB_Analysis')
# BERTopic uses tokenisation so throws warnings if multiprocessing after
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

combined = pd.read_csv("data/mergedDataDec.csv", encoding='utf-8').map(str).map(str.strip).reset_index(drop=True)

# combine title, keywords, abstract, relevance and trace type columns if they are not nan or empty into a single column
combined.index
cols = ['Title', 'Trace_Type','Study_Type', 'Keywords', 'Abstract', 'Exp_Conditions_and_Results',
    'Relevance_to_Canada', 'Addressed question',
    'Activity context', 'Category', 'Specifications',
    'Variables of interest', 'stringency of control', 'No of individuals',
    'Replicates per Individual and condition', 'Nucleic Acid',
    'Bodily origin', 'depositor characteristics',
    'Criteria for shedder status', 'Previous activities',
    'Contact scenario', 'Primary substrate type',
    'Primary substrate Material', 'Deposit', 'Delay (conditions)',
    'Secondary substrate type', 'Secondary Substrate material',
    'Type of secondary contact', 'Further transfer',
    'Background DNA on sampled surface', 'Sampling time',
    'Persistance (conditions)', 'Sampling method', 'Sampling area',
    'Extraction', 'DNA Quantification', 'Input for Profiling', 'Profiling',
    'Reference samples', 'Profile interpretation and mixture analysis',
    'RNA data interpretation', 'DNA Quantitiy', 'Profile Quality',
    'Parameter used for comparison', 'Summary of results',
    'Raised questions (by authors)', 'Cautionary remarks', 'author_keywords']

combined['allData'] = combined[cols].apply(uf._join_non_na, axis=1)

# The sentences to encode
dataAsList = combined['allData'].to_list()

combined.to_csv("modelReadyData.csv")

#set domain specific stop words (ie forensic, bayesian etc) as too general. At the trace type level we dont care about methods,
# within trace type we can look at these
forensic_stopwords = [
    'forensic', 'bayesian', 'analysis','samples', 'analyses', 'sampled', 'bayes', 'thereom',
    'forensics', 'evidence', 'examination', 'investigation', 'investigations',
    'investigator','sample', 'examined', 'method', 'methods', 'methodology',
    'investigated', 'investigate', 'laboratory', 'laboratories', 'apprehending',
    'research', 'researches', 'case', 'cases', 'casework', 'caseworks', 'testing',
    'examining', 'evaluated', 'evaluation', 'evaluates', 'assessed', 'assessment',
    'crime', 'probabilistic', 'probability', 'probabilities', 'policing'
    'interpretation', 'likelihood', 'ratio','collected','analyze', 'experiments', 
    'analyse', 'experiment', 'analyzed','specimens', 'examiners','analyzing', 'findings', 
    'propositions', 'study', 'techniques', 'technique', 'instrumentation', 'instruments',
    'measurements', 'measurement', 'validation', 'validated', 'validating', 'swabs',
    'swab', 'replicates', 'forensically', 'data', 'results', 'result', 'using', 'used', 
    'use', 'based', 'different', 'assess', 'test', 'tests', 'metholodogies',
    'analysed', 'tested', 'detection', 'detected', 'detect', 'compare',
    'compared', 'comparison', 'comparisons', 'identified', 'identification', 
    'identify','identifies', 'reviewed', 'review', 'reviews', 'obtained', 'obtain',
    'assessing', 'investigations', 'conclusions', 'conclusion', 'concluded',
    'deposited', 'swabbing', 'swabbed', 'studies', 'investigative', 'examines',
    'police', 'officer', 'officers', 'detecting', 'evaluate', 'determine',
    'determining', 'collecting', 'collection', 'analyzes', 'methodologies', 'examine',
    'screening','analysing', 'examinations','evaluating', 'evaluations', 'observations',
    'comparative', 'comparatively', 'detects', 'determined', 'determines', 'investigators',
    'investigates','measure', 'measured', 'measures', 'studied', 'analytical', 'differences'
    'validations', 'validates', 'utilized', 'utilize', 'utilizes', 'documented'
    'characteristics', 'recommendations','factors', 'consideration', 'considerations',
    'investigaating', 'probative','investigating', 'characteristic', 'lab', 'utilizing', 
    'usefulness', 'characterisation', 'characterize', 'characterized', 'characterization', 
    'fbi', 'law enforcement', 'crimes', 'law', 'enforcement', 'practices', 'practice', 
    'caratristiques', 'practise', 'practises', 'security', 'considered', 'consider', 
    'conducted', 'conduct', 'conducts', 'conduction', 'derives', 'derived', 'deriving', 'employed', 
    'employs', 'employing', 'sciences', 'science', 'implementation', 'implementing', 'implemented', 
    'implement', 'involving', 'involved', 'involves', 'utilization', 'utilisations', 'labwork', 
    'laboratorywork', 'laboratoryworks', 'specialized', 'specialise', 'specialises', 'specialised', 
    'practitioner', 'practitioners', 'practising', 'practised', 'authorities', 'authority', 
    'authoritarian', 'regarding', 'regard', 'regards', 'enfsi', 'interpol', 'obtaining', 
    'obtains', 'hypothesis', 'hypotheses', 'evidential', 'evidentially', 'evidences', 
    'operational', 'operations', 'operation', 'operates', 'operate', 'operating', 'standards', 
    'standard', 'standardization', 'standardisations', 'standardise', 'standardises', 
    'standardized', 'protocols', 'protocol', 'procedures', 'procedure', 'procedural', 
    'procedurally', 'practiced', 'criminal', 'criminial', 'criminology', 'criminological',
    'justice', 'search', 'seizure', 'admissibility', 'admissible', 'technology', 'exploitation',
    'exploiting', 'exploited', 'exploits', 'explored', 'normal', 'prepared', 'prepares', 
    'preparing', 'preparation', 'preparations', 'samples analyzed', 'interpretation forensic', 
    'forensic context', 'evaluation forensic', 'feature classification'
    ]

extra_forensic_stopwords = [
    'scene', 'scenes', 'extracted', 'spectrometry', 'extraction', 'chromatography', 
    'analyzer', 'profiling', 'assay', 'assays', 'sampling', 'well', 'analyser', 
    'profile', 'profiles', 'quantification', 'quantified', 'quantify', 'markers',
    'marker', 'spectrometer', 'sequencing', 'sequenced', 'sequencer', 'amplification', 
    'amplified', 'packaging', 'amplify', 'loci', 'locus', 'electrophoresis', 
    'electrophoretic'
                            ]
                      
# Get the list of other language stop words
multilingual_stop_words = stopwords.words()

custom_stopwords = list(text.ENGLISH_STOP_WORDS.union(
    forensic_stopwords,
    # extra_forensic_stopwords,
    multilingual_stop_words)
                        )

preprocessed_docs = [uf.preprocess(doc) for doc in dataAsList]

# create vectorise method
vectorizer_model = CountVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1, 2),
    min_df=0.2,
    max_df=0.6
)


# Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", 
    prompts={
        "classification": "Classify the following text into topics relating to forensic sample types: ",
        "retrieval": "Retrieve semantically similar text, input is in multiple languages: ",
        "clustering": "Identify the topic or theme of forensic sample type or discipline based on the provided \
            text. Do not include law enforcement, justice, statistics or crime as themes. Label topic with trace type or originating location"
    })


# Calculate embeddings by calling model.encode() saves time later for BERTopic to avoid doing this internally
embeddings = model.encode(dataAsList)
print(embeddings.shape)
# should be same rows as data ie [3279, 384]

'''
Topic Modeling with BERTopic: Minimum Viable Example
References:
[1] https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html
[2] https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html
[3] https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html
'''

# Fine-tune the topic representations
representation_model = KeyBERTInspired(
    top_n_words=100,
    nr_repr_docs=800,
    nr_samples=2000,
    nr_candidate_words=2000)

# # create a seed topic list to grow from
# topic_list=[['paints', 'paint','pigments','pigment',
#      'colour','coating','coatings'],
# ['glass', 'glasses','refractive','windshields'],
# ['arson','petrol','gasoline', 'fires','hydrocarbons',
#      'fire', 'liquids', 'ignitable'],
# ['methamphetamine','methadone','cocaine','drug', 'drugs',
#      'metabolites','amphetamine','heroin','metabolite'],
# ['lipstick','cosmetics','lipsticks','cosmetic','lip','glitter'],
# ['pollen','botany','vegetation','spores','palynological','palynology','plants'],
# ['fingerprints','fingerprint','fingermarks','fingermark','ngerprints'],
# ['biomarkers','rna','mrna','luminol','haemoglobin'],
# ['bloodstain','patterns','blood','pattern','arterial','bpa'],
# ['hairs','hair'],
# ['dna','contamination', 'contaminations'],
# ['remains','skeletal','bones','cadavers','teeth',
#       'dental','tooth','dentine','dentistry', 'dentin'],
# ['sperm','spermatozoa','semen','seminal', 'postcoital','vaginal'],
# ['soils','soil','dust','mineralogy','mineralogical',
#       'geology','mineral','geoscience'],
# ['dna','mixture','mixtures'],
# ['condom','condoms','lubricants','lubricant',
#        'condomlubricants', 'lubricated'],
# ['tape','tapes','adhesive'],
# ['fingernails','nails','fingernail','nail','scratching'],
# ['saliva','salivary','mouth'],
# ['fibres','fabrics','textile','cotton','fibre','fibers', 
#       'garments', 'fabric','nylon','wool','dyed','polyester','shirt',
#       'garment','fiber','textile', 'cloth','clothing','clothes',
#      'material'],
# ['cartridge','ammunition','ammunitions','bullet','cartridges'],
# ['laundry','washing'],
# ['dna','wearer','scalp','trace','touch','skin','hands','touched'],
# ['gsr','gunshot','gunpowder','elemental','primer','residue',
#       'powder','residue','residues','compounds'],
# ['explosive','explosives','bombs','bomb','chemical'],
# ['firearm','handguns','rifles',
#      'pistol','shotguns','gun','shotgun'],
# ['corde','rope']]

# macOS has a bug with matrix multiplication that causes runtime warnings of zero division. 
    # Create custom HDBSCAN model
hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    min_samples=1,
    cluster_selection_epsilon=0.1,
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



# load optimised model
# best_model=BERTopic.load("bertopic_optimized_hdbscan", embedding_model=model)


with open("best_model_params.json", "r") as f:
    best_results = json.load(f)

combined['T1_topic'] = new_topics #best_results[1]

#combined['T1_probs'] = best_results[2] # type: ignore

topic_info = topic_model.get_topic_info() # type: ignore
#topic_model.set_topic_labels(list(topic_info['Name']))

# Exclude the -1 topic (outliers) for labeling the main topics
topic_info = topic_info[topic_info['Topic'] != -1].reset_index(drop=True)

# associate colours with the topics
colours_list = [
    '#d62728','#f3722c','#ffa15a','#fecb52','#f5f511',
    '#b6e880','#2ca02c','#4d908e','#43aa8b','#00cc96','#4cddc9',
    '#90dbf4','#19d3f3','#005073','#718591','#1f77b4','#a3c4f3','#cfbaf0',
    '#6a19b5','#636efa','#ab63fa','#b9fbc0','#ff97ff','#ffcfd2'
    ]

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


min_dist = 2.5 # minimum allowed distance between any label and any other object

positions = uf.resolve_overlaps(positions, coords, min_dist, repulsion_strength=0.5,
        marker_repulsion_strength=1, attraction_strength=0.08, max_step=2)

offset_x, offset_y = positions[:, 0], positions[:, 1]

# Add static labels as annotations to the Plotly figure
annotations = []
for index, row in topic_info.iterrows():

    annotations.append(
        dict(
            ax=offset_x[index], # arrow tail pos # type: ignore
            ay=offset_y[index], # type: ignore
            text=f"Topic {index}", # Use topic name
            showarrow=True,
            x=coords[index][0], #arrow head pos
            y=coords[index][1],
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
    marker=dict(color=mapped_colours ),
    selector=dict(mode='markers'),
    # name=topic_info['Name'].tolist()
)

# update to add the static labels
dist_map.update_layout(
    annotations=annotations,               
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
    }
    )

# Display the figure
dist_map.show()

displayTable = topic_info[['Topic','Name','Count']]
displayTable.to_html(index=False)

# topic-terms barcharts
bar_fig = topic_model.visualize_barchart(top_n_topics=26, autoscale=True, width=350) # type: ignore
bar_fig.show()

hierarchical_topics = topic_model.hierarchical_topics(dataAsList) # type: ignore

# topics dendrogram
hierarch_fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics) # type: ignore
hierarch_fig.show()



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