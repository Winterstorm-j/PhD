#! /Users/jbuc045/Projects/.venv/bin/python

import pandas as pd
import numpy as np
# import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import re
import ast

def get_names(val,col):
    if isinstance(val, str):
        try:
            # Use literal_eval to convert the string to a list of dicts
            data = ast.literal_eval(val)
            
            # Use a list comprehension to extract 'display_name' from each dictionary
            return [d[col] for d in data if col in d]
        except (ValueError, SyntaxError):
            # Handle malformed strings
            return val
    return val


####### TTADB
# IMPORT TTADB removing any trailing whitespace
TPPR_db = pd.read_csv("PhD/TPPRDB_Analysis/TTADB-Aug2025.csv").map(str).map(str.strip).reset_index(drop=True)

# remove empty columns and rename last column
TPPR_db = TPPR_db.drop(['Unnamed: 13','Unnamed: 14', 'Unnamed: 15'], axis=1)
TPPR_db.rename(columns={'Unnamed: 16': 'Citation'}, inplace=True)

# split full citationin into Authors (words before the date, Year numbers between brackets, Title everything after) 
searchTerm = [re.search(r'(.+)\(([0-9sd\.]{4})\)(.*)',row) if row else () for row in TPPR_db['Authors']]
TPPR_db['Authors'] = [row.groups()[0] if row else None for row in searchTerm]
TPPR_db['Year'] = [row.groups()[1] if row else None for row in searchTerm]
TPPR_db['Title'] = [row.groups()[2] if row else None for row in searchTerm]

# remove leading whitespace and fullstop then split into columns on remaining fullstops, 
# Title everything before fullstop, Journal,Book,Meeting everything after (with leading whitespace removed)
TPPR_db['Title'] = TPPR_db['Title'].str.replace(r"^[ ]*\.[ ]*", "", regex=True)
searchTitle = TPPR_db['Title'].str.split(r"\.",expand=True)
TPPR_db['Title'] = searchTitle[0]
TPPR_db['Journal_Book_Institution_Meeting'] = searchTitle[1] + searchTitle[2] + searchTitle[3]
TPPR_db['Journal_Book_Institution_Meeting'] = TPPR_db['Journal_Book_Institution_Meeting'].str.replace(r"^[ ]*", "", regex=True)

# initialise empty output lists
Publishing_Details = []
Journal = []
 
# split Journal_Book_Institution_Meeting into the two output lists on first comma, 
# If it starts with Academic Press, it is a book so everything to Publishing_Details only
# save back to TPPR_db object
for row in TPPR_db['Journal_Book_Institution_Meeting'] :
    if (row and str(row).startswith('Academic Press')):
        Publishing_Details.append(row)
        Journal.append(None)
    elif (not row or pd.isna(row)):
        Publishing_Details.append(None)
        Journal.append(None)
    else: 
        splitCol = str(row).split(",", maxsplit=1)
        if len(splitCol)==2:
            Journal.append(splitCol[0])
            Publishing_Details.append(splitCol[1])
        else:
             Journal.append(splitCol[0])
             Publishing_Details.append(None)
        
TPPR_db['Journal_Book_Institution_Meeting'] = pd.Series(Journal)
TPPR_db['Publishing_Details'] = pd.Series(Publishing_Details).str.replace(r"^[ ]*", "", regex=True)


####### DNA-TrAC
# IMPORT DNA-TRAC removing any trailing whitespace
dnaTrac = pd.read_excel('PhD/TPPRDB_Analysis/DNA-TrAC_Ver-2019-12-16.xlsx').map(str).map(str.strip).reset_index(drop=True)

#remove duplicates
newDnaTrac = dnaTrac.groupby(['Title','Authors','Year']).agg(pd.unique).apply(lambda x: x[0] if len(x)==1 else x)

#unlist all columns imported as lists
newDnaTrac['Journal'] = newDnaTrac['Journal'].apply(lambda x: x[0])
[df[col].apply(lambda x: np.array2string(x)) if isinstance(df[col], np.ndarray) else df[col] for col in df.columns ]


###### Web of Science results
#IMPORT WoS SEARCH RESULTS removing any trailing whitespace
searchResults = pd.read_excel('PhD/TPPRDB_Analysis/articleList.xlsx').map(str).map(str.strip).reset_index(drop=True)

#remove duplicates
searchResults = searchResults.loc[~searchResults.duplicated(subset='uid'),:]


newTTADB = TPPR_db.sort_values(['Title','Authors','Year','Relevance_to_Canada'], na_position='last').groupby(
    ['Title','Authors','Year']).agg(pd.unique).apply(lambda x: x[0] if len(x)==1 else x)

# change results so all dictionary columns contain flat data
searchResults = searchResults.replace({"nan": np.nan })
searchResults['authors'] = searchResults.iloc[:,10].apply(lambda x: get_names(x,'wos_standard')).apply(lambda x: "; ".join(x) if isinstance(x,list) else x )
searchResults['book_editors'] = searchResults.iloc[:,13].apply(lambda x: get_names(x,'display_name')).apply(lambda x: "; ".join(x) if isinstance(x,list) else x )
searchResults['editors'] = searchResults.iloc[:,13].apply(lambda x: get_names(x,'display_name')).apply(lambda x: "; ".join(x) if isinstance(x,list) else x )
searchResults['pages'] = searchResults.iloc[:,13].apply(lambda x: get_names(x,'range')).apply(lambda x: "- ".join(x) if isinstance(x,list) else x )

# rename columns in searchResulat to match the other datasets
searchResults.columns = ['Title', 'source_title', 'Year', 'Month', 'Volume', 'Issue', 'Supplement', 'special_issue', 'article_number', 'pages',
'Authors', 'inventors', 'book_corp', 'book_editors', 'books', 'additional_authors', 'anonymous', 'assignees', 'Editors', 'record',
'references', 'related', 'doi', 'issn', 'eissn', 'isbn', 'eisbn','pmid', 'author_keywords', 'unique_type', 'uid']

searchResults = searchResults.reset_index(drop=True)
searchResults.shape
newTTADB.shape
newTTADB.head(1)


dnaTrac.shape
newDnaTrac.columns

#combine data sources into one df making sure case is not a factor ()
# DNA_trac to TTADB
combined = newTTADB.map(str).map(str.upper).merge(
    newDnaTrac.map(str).map(str.upper),
          on=['Title','Authors','Year'],suffixes = ['_dnatrac', '_ttadb'], validate='one_to_one', how="outer").reset_index(drop=False)

combined.shape
combined.loc[~pd.isna(combined.Journal),'Journal']

# combine WoS search reaults to combined
combined = combined.map(str).map(str.upper).merge(
    searchResults.map(str).map(str.upper),
          on=['Title','Authors','Year'],suffixes =['_comb', '_wos'], validate='one_to_one', how="outer").reset_index(drop=False)


combined.shape
combined.columns
searchResults.columns


# reset format of all character strings to title case 
for col in combined.select_dtypes(include='object').columns:
    combined[col] = combined[col].str.title()

combined.to_csv('PhD/TPPRDB_Analysis/mergedDataSept.csv')#, encoding='ISO-8859-2')

# combine title, keywords, abstract, relevance and trace type columns into a single column
combined.index
combined['allData'] = combined[['Title','Trace_Type','Keywords','Abstract','Exp_Conditions_and_Results','Relevance_to_Canada']].agg('; '.join, axis=1)

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
topic_model.visualize_documents(dataAsList, reduced_embeddings=np.array(reduced_embeddings))#.write_html("./PhD-Windows/TPPRDB_Analysis/reduced_projections.html")

plt.hist(dataAsList, s=50, linewidth=0, c='b', alpha=0.25)



# time series analysis

# The data to encode
datedData = TPPR_db[['Year','Title','Trace_Type','Keywords','Abstract','Exp_Conditions_and_Results','Relevance_to_Canada']]
dataAsDatedList = datedData.apply(
    lambda row: '; '.join(row.dropna().astype(str)), axis=1
).to_list()

date = datedData.Year
date[date=='s.d.'] = np.NaN


topics_over_time = topic_model.topics_over_time(dataAsDatedList, date.astype('str').to_list())
model.visualize_topics_over_time(topics_over_time, topics=[range(1,21)])



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