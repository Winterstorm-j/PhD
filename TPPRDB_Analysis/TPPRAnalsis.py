#! .venv/bin/python3
 
import os
os.chdir('./PhD/TPPRDB_Analysis')

import pandas as pd
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
import util_functions as uf
from nltk.corpus import stopwords
import plotly.io as pio
pio.renderers.default = "browser"

combined = pd.read_csv("data/mergedDataSept.csv", encoding='utf-8').map(str).map(str.strip).reset_index(drop=True)

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

#set domain specific stop words (ie forensic, bayesian etc) as too general. At the trace type level we dont care about methods,
# within trace type we can look at these
forensic_stopwords = ['forensic', 'bayesian', 'analysis','samples', 'analyses', 'sampled',
                      'forensics', 'evidence', 'examination', 'investigation', 'investigations',
                      'investigator','sample', 'examined', 'method', 'methods', 'methodology',
                      'investigated', 'investigate', 'laboratory', 'laboratories',
                      'research', 'researches', 'case', 'cases', 'casework', 'caseworks', 'testing',
                      'examining', 'evaluated', 'evaluation', 'evaluates', 'assessed', 'assessment',
                      'crime', 'probabilistic', 'probability', 'probabilities', 'scene', 'scenes',
                      'interpretation', 'likelihood', 'ratio','collected','analyze', 'experiments', 
                      'analyse', 'experiment', 'analyzed','extracted','specimens', 
                      'spectrometry', 'extraction', 'examiners','analyzing', 'findings', 'propositions', 
                      'chromatography', 'study','analyzer', 'profiling', 'assay', 'assays', 'techniques',
                      'technique', 'instrumentation', 'instruments', 'sampling', 'measurements', 
                      'measurement', 'validation', 'validated', 'validating', 'swabs', 'swab', 
                      'sequencing', 'sequenced', 'sequencer', 'amplification', 'amplified', 
                      'amplify', 'loci', 'locus', 'electrophoresis', 'electrophoretic', 'replicates',
                      'forensically', 'data', 'results', 'result', 'using', 'used', 'use', 'based', 
                      'different', 'assess', 'test', 'tests', 'well', 'metholodogies',
                    'analysed', 'analyser', 'tested', 'detection', 'detected', 'detect', 'compare',
                    'compared', 'comparison', 'comparisons', 'identified', 'identification', 
                    'identify','identifies', 'reviewed', 'review', 'reviews', 'obtained', 'obtain',
                    'assessing', 'investigations', 'conclusions', 'conclusion', 'concluded',
                    'deposited', 'swabbing', 'swabbed', 'studies', 'investigative', 'examines',
                    'profile', 'profiles', 'quantification', 'quantified', 'quantify', 
                    'markers', 'marker', 'police', 'officer', 'officers', 'detecting', 'evaluate', 'determine',
                    'determining', 'collecting', 'collection', 'analyzes', 'methodologies', 'examine',
                    'screening','analysing', 'examinations','evaluating', 'evaluations', 'observations',
                    'comparative', 'comparatively', 'detects', 'determined', 'determines', 'investigators',
                    'investigates','measure', 'measured', 'measures', 'studied', 'analytical', 'differences'
                    'validations', 'validates', 'utilized', 'utilize', 'utilizes', 'documented'
                    'spectrometer', 'characteristics', 'recommendations','factors', 
                    'consideration', 'considerations','investigaating', 'probative',
                    'investigating', 'characteristic', 'lab', 'utilizing', 'usefulness', 'packaging',
                    'characterisation', 'characterize', 'characterized', 'characterization', 'fbi', 
                    'law enforcement', 'crimes', 'law', 'enforcement', 'practices', 'practice', 'caratristiques',
                    'practise', 'practises', 'security', 'considered', 'consider', 'conducted', 'conduct', 'conducts',
                    'conduction', 'derives', 'derived', 'deriving', 'employed', 'employs', 'employing', 'sciences', 'science'
                    'implementation', 'implementing', 'implemented', 'implement', 'involving', 'involved', 'involves',
                    'utilization', 'utilisations', 'labwork', 'laboratorywork', 'laboratoryworks', 'specialized',
                    'specialise', 'specialises', 'specialised', 'practitioner', 'practitioners', 'practising', 'practised', 
                    'authorities', 'authority', 'authoritarian', 'regarding', 'regard', 'regards', 'enfsi', 'interpol',
                    'obtaining', 'obtains', 'hypothesis', 'hypotheses',
                    'evidential', 'evidentially', 'evidences', 'operational', 'operations', 'operation',
                    'operates', 'operate', 'operating', 'standards', 'standard', 'standardization', 'standardisations',
                    'standardise', 'standardises', 'standardized', 'protocols', 'protocol', 'procedures', 'procedure',
                    'procedural', 'procedurally', 'practiced', 'criminal', 'criminial', 'criminology', 'criminological', 
                    'justice', 'search', 'seizure', 'admissibility', 'admissible', 'technology', 'exploitation', 'exploiting',
                    'exploited', 'exploits', 'explored', 'normal', 'prepared', 'prepares', 'preparing', 'preparation', 
                    'preparations'
                      ]

# Get the list of other language stop words
multilingual_stop_words = stopwords.words()

custom_stopwords = list(text.ENGLISH_STOP_WORDS.union(forensic_stopwords,multilingual_stop_words))

preprocessed_docs = [uf.preprocess(doc) for doc in dataAsList]

# create vectorise method
vectorizer_model = CountVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1, 2),
    min_df=0.2,
    max_df=0.6
)


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", 
    prompts={
        "classification": "Classify the following text into topics relating to forensic sample types: ",
        "retrieval": "Retrieve semantically similar text, input is in multiple languages: ",
        "clustering": "Identify the topic or theme of forensic sample type or discipline based on the text: ",
    })


# 2. Calculate embeddings by calling model.encode() not needed as BERTopic will do this internally
# embeddings = model.encode(dataAsList)
# print(embeddings.shape)
# should be same dims as data ie [2512, 384]


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

# Fine-tune the topic representations
representation_model = KeyBERTInspired(
    top_n_words=100,
    nr_repr_docs=800,
    nr_samples=2000,
    nr_candidate_words=2000)

# Clustering model: See [2] for more details
cluster_model = HDBSCAN(min_cluster_size = 5, 
                        metric = 'euclidean', 
                        cluster_selection_method = 'eom', 
                        prediction_data = True)

# create a seed topic list to grow from
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

# macOS has a bug with matrix multiplication that causes runtime warnings of zero division. 

# Bertopic model instantiation
topic_model = BERTopic(vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    embedding_model = model,
    hdbscan_model = cluster_model, 
    # nr_topics=30,
    seed_topic_list=topic_list)

# Fit the model on a corpus
topics, probs = topic_model.fit_transform(dataAsList)

topic_info = topic_model.get_topic_info()
topic_info['Name']
# Save intertopic distance map as HTML file
dist_map = topic_model.visualize_topics(width =1000, height=800)
#.write_html("./PhD-Windows/TPPRDB_Analysis/intertopic_dist_map.html")
dist_map.show()

# Save topic-terms barcharts as HTML file
bar_fig = topic_model.visualize_barchart(top_n_topics = 100)
#.write_html("./PhD-Windows/TPPRDB_Analysis/barchart.html")
bar_fig.show()

hierarchical_topics = topic_model.hierarchical_topics(dataAsList)

# Save topics dendrogram as HTML file
hierarch_fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
hierarch_fig.show()#.write_html("./PhD-Windows/TPPRDB_Analysis/hieararchy.html")




# Save documents projection as HTML file
visualise_docs = topic_model.visualize_documents(docs=dataAsList, topics=topics)
#.write_html("./PhD-Windows/TPPRDB_Analysis/projections.html")
visualise_docs.show()

# # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
# reduced_embeddings = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine').fit_transform(embeddings)
# reduced_topics = topic_model.visualize_documents(dataAsList, reduced_embeddings=np.array(reduced_embeddings))
# #.write_html("./PhD-Windows/TPPRDB_Analysis/reduced_projections.html")
# reduced_topics.show()




# time series analysis

# The data to encode
datedData = combined[['Year','Title','Trace_Type','Keywords','Abstract','Exp_Conditions_and_Results','Relevance_to_Canada']]
dataAsDatedList = datedData.apply(
    lambda row: '; '.join(row.dropna().astype(str)), axis=1
).to_list()

date = datedData.Year
date[date=='s.d.'] = np.nan


topics_over_time = topic_model.topics_over_time(dataAsDatedList, date.astype('str').to_list())
# ßmodel.visualize_topics_over_time(topics_over_time, topics=[range(1,21)])



''' 
'Index', 'Doc_Type', 'Authors', 'Year', 'Title','Journal_Book_Institution_Meeting', 'Publishing_Details', 'Trace_Type',
'Study_Type', 'Keywords', 'Abstract', 'Exp_Conditions_and_Results','Relevance_to_Canada'

'Column1', 'source_title', 'publish_year', 'publish_month', 'volume', 'issue', 'supplement', 'special_issue', 'article_number', 'pages',
'authors', 'inventors', 'book_corp', 'book_editors', 'books', 'additional_authors', 'anonymous', 'assignees', 'editors', 'record',
'references', 'related', 'doi', 'issn', 'eissn', 'isbn', 'eisbn','pmid', 'author_keywords', 'unique_type', 'uid'

'Authors', 'Year', 'Title', 'Journal', 'Addressed question','Activity context', 'Category', 'Specifications','Variables of interest',
'stringency of control', 'No of individuals','Replicates per Individual and condition', 'Nucleic Acid','Bodily origin', 'depositor characteristics',
'Criteria for shedder status', 'Previous activities','Contact scenario', 'Primary substrate type',
'Primary substrate Material', 'Deposit', 'Delay (conditions)','Secondary substrate type', 'Secondary Substrate material',
'Type of secondary contact', 'Further transfer','Background DNA on sampled surface', 'Sampling time','Persistance (conditions)', 
'Sampling method', 'Sampling area','Extraction', 'DNA Quantification', 'Input for Profiling', 'Profiling',
'Reference samples', 'Profile interpretation and mixture analysis','RNA data interpretation', 'DNA Quantitiy', 'Profile Quality',
'Parameter used for comparison', 'Summary of results','Raised questions (by authors)', 'Cautionary remarks'


'Title', 'Year', 'Index', 'Doc_Type', 'Authors', 'Journal_Book_Institution_Meeting', 'Publishing_Details', 'Trace_Type',
'Study_Type', 'Keywords', 'Abstract', 'Exp_Conditions_and_Results', 'Relevance_to_Canada', 'Journal', 'Addressed question',
'Activity context', 'Category', 'Specifications', 'Variables of interest', 'stringency of control', 'No of individuals',
'Replicates per Individual and condition', 'Nucleic Acid', 'Bodily origin', 'depositor characteristics',
'Criteria for shedder status', 'Previous activities', 'Contact scenario', 'Primary substrate type',
'Primary substrate Material', 'Deposit', 'Delay (conditions)', 'Secondary substrate type', 'Secondary Substrate material',
'Type of secondary contact', 'Further transfer', 'Background DNA on sampled surface', 'Sampling time',
'Persistance (conditions)', 'Sampling method', 'Sampling area', 'Extraction', 'DNA Quantification', 'Input for Profiling', 'Profiling',
'Reference samples', 'Profile interpretation and mixture analysis', 'RNA data interpretation', 'DNA Quantitiy', 'Profile Quality',
'Parameter used for comparison', 'Summary of results', 'Raised questions (by authors)', 'Cautionary remarks'
'''

# evaluate topic model for coherence and diversity
coherence_score = uf.calculate_coherence_score(topic_model, dataAsList)

diversity_score = uf.calculate_diversity_score(topic_model)

print(f"Coherence Score: {coherence_score}")
print(f"Diversity Score: {diversity_score}")

# Save the model
# topic_model.save("models/TPPRDB_BERTopic_Model")    
