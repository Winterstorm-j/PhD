#! /Users/jbuc045/Projects/.venv/bin/python
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
from sklearn.feature_extraction.text import CountVectorizer
import re
import ast
from util_functions import get_names, combine_group_rows, preprocess_string_columns, _extract_range, preprocess, _join_non_na

####### TTADB
# IMPORT TTADB removing any trailing whitespace
TPPR_db = pd.read_csv("TTADB-Aug2025.csv", encoding='utf-8').map(str).map(str.strip).reset_index(drop=True)

# remove empty columns and rename last column
TPPR_db = TPPR_db.drop(['Unnamed: 13','Unnamed: 14', 'Unnamed: 15'], axis=1)
TPPR_db.rename(columns={'Unnamed: 16': 'Citation'}, inplace=True)

# split full citationin into Authors (words before the date, Year numbers between brackets, Title everything after) 
searchTerm = [re.search(r'(.+)\(([0-9sd\.]{4})\)(.*)',row) if row else () for row in TPPR_db['Authors']]
TPPR_db['Authors'] = [row.groups()[0] if row else None for row in searchTerm]
TPPR_db['Year'] = [row.groups()[1] if row else None for row in searchTerm]
TPPR_db['Title'] = [row.groups()[2] if row else None for row in searchTerm]

#standardise how authors are displayed
TPPR_db['Authors'] = (TPPR_db['Authors'].str.replace(r"[;,]", " ", regex=True)
                      .str.replace(r"\.", "", regex=True)
                      .str.replace(r"\s+", " ", regex=True)
                      .str.strip()
)

# remove leading whitespace and fullstop then split into columns on remaining fullstops, 
# Title = everything before fullstop with «, " , and » removed, 
TPPR_db['Title'] = TPPR_db['Title'].str.replace(r"^\s*\.\s*", "", regex=True)

# remove leading whitespace and fullstop then split into columns on the first fullstop followed by a space or ', in '
searchTitle = TPPR_db['Title'].str.split(r"\.\s(?=[A-Z])|,\s*in\s", n=1, expand=True)
TPPR_db['Title'] = searchTitle[0]
TPPR_db['Title'] = TPPR_db['Title'].str.replace(r"[«\"»]*", "", regex=True)

# Journal,Book,Meeting = everything after (with leading whitespace removed)
TPPR_db['Journal_Book_Institution_Meeting'] = searchTitle[1]
TPPR_db['Journal_Book_Institution_Meeting'] = TPPR_db['Journal_Book_Institution_Meeting'].str.replace(r"^\s*", "", regex=True)

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
        splitCol = re.split(r",|\.",str(row), maxsplit=1)
        if len(splitCol)==2:
            Journal.append(splitCol[0])
            Publishing_Details.append(splitCol[1])
        else:
             Journal.append(splitCol[0])
             Publishing_Details.append(None)
        
TPPR_db['Journal_Book_Institution_Meeting'] = pd.Series(Journal).str.strip()
TPPR_db['Publishing_Details'] = pd.Series(Publishing_Details).str.strip()

newTTADB = combine_group_rows(TPPR_db, ['Title','Authors','Year'])

####### DNA-TrAC
# IMPORT DNA-TRAC removing any trailing whitespace
dnaTrac = pd.read_csv('DNA-TrAC_Ver-2019-12-16.csv', dtype=str, encoding='utf-8')

dnaTrac = dnaTrac.map(str).map(str.strip).reset_index(drop=True)

dnaTrac['Authors'] = (dnaTrac['Authors'].str.replace(r"[;,]", " ", regex=True)
                      .str.replace(r"\.", "", regex=True)
                      .str.replace(r"  ", " ", regex=True)
)

dnaTrac['Journal'] = dnaTrac['Journal'].str.replace(r"FSI[:]* ", "FORENSIC SCIENCE INTERNATIONAL: ", regex=True)
dnaTrac['Journal'] = dnaTrac['Journal'].str.replace("FSI", "FORENSIC SCIENCE INTERNATIONAL")
dnaTrac['Journal'] = dnaTrac['Journal'].str.replace("Science and Justice", "SCIENCE & JUSTICE")

newDnaTrac = combine_group_rows(dnaTrac, ['Title', 'Authors', 'Year'])

# find duplicated rows in newDnaTrac in title, authors, and year
duplicates = newDnaTrac[newDnaTrac.duplicated(subset=['Title', 'Authors', 'Year'], keep=False)]

newTTADB.shape
newTTADB.head(1)
dnaTrac.shape
# newDnaTrac.loc[33,"Cautionary remarks"]
newTTADB['Journal_Book_Institution_Meeting']

# Combine data sources into one df making sure case is not a factor in matching
preprocessed_ttadb = preprocess_string_columns(newTTADB.copy())
preprocessed_dnatrac = preprocess_string_columns(newDnaTrac.copy())

# Find and display non-unique merge keys before merging
ttadb_dupes = preprocessed_ttadb[preprocessed_ttadb.duplicated(subset=['Title', 'Authors', 'Year'], keep=False)]
dnatrac_dupes = preprocessed_dnatrac[preprocessed_dnatrac.duplicated(subset=['Title', 'Authors', 'Year'], keep=False)]

print(ttadb_dupes[['Title', 'Authors', 'Year']])
print(dnatrac_dupes[['Title', 'Authors', 'Year']])

combined = preprocessed_ttadb.merge(
    preprocessed_dnatrac,
    on=['Title', 'Authors', 'Year'],
    suffixes=['_dnatrac', '_ttadb'],
    validate='one_to_one',
    how="outer"
).reset_index()

combined.shape
combined['Journal_Book_Institution_Meeting'] = [
    combined.loc[record,'Journal'] 
                if (pd.isna(combined.loc[record,'Journal_Book_Institution_Meeting']) or combined.loc[record,'Journal_Book_Institution_Meeting'] == None) 
                else combined.loc[record,'Journal_Book_Institution_Meeting'] for record in combined.index
                ]

# sense check - if Journal_Book... from TTADB is different to Journal in DNA TrAc 
test = combined.loc[(combined['Journal_Book_Institution_Meeting'] != combined['Journal']) & 
                    (~pd.isna(combined['Journal'])),['Title','Authors','Year','Journal_Book_Institution_Meeting', 'Journal']]

# replace Journal_Book... with Journal from DNA Trac if journal name has been truncated due to previous split
combined.loc[(combined['Journal_Book_Institution_Meeting'] != combined['Journal']) & 
             (~pd.isna(combined['Journal'])),'Journal_Book_Institution_Meeting'] = test['Journal']

###### Web of Science results
#IMPORT WoS SEARCH RESULTS removing any trailing whitespace
searchResults = pd.read_csv('articleList.csv', encoding='utf-8').map(str).map(str.strip).reset_index(drop=True)

#remove duplicates
searchResults = searchResults.loc[~searchResults.duplicated(subset='uid'),:]

# change results so all dictionary columns contain flat data
searchResults = searchResults.replace({"nan": np.nan })

searchResults['authors'] = (
    searchResults.iloc[:, 10]
    .apply(lambda x: str(x) if isinstance(x, float) else x)
    .apply(lambda x: get_names(x, 'wos_standard'))
    .apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    .str.replace(r"[;,]", " ", regex=True)
    .str.replace(r"\.", "", regex=True)
    .str.replace(r"  ", " ", regex=True)
)
searchResults['book_editors'] = (
    searchResults.iloc[:,13]
    .apply(lambda x: str(x) if isinstance(x, float) else x)
    .apply(lambda x: get_names(x,'display_name'))
    .apply(lambda x: " ".join(x) if isinstance(x,list) else x )
    .str.replace(r"[;,]", " ", regex=True)
    .str.replace(r"\.", "", regex=True)
    .str.replace(r"  ", " ", regex=True)
)
searchResults['editors'] = (
    searchResults.iloc[:,17]
    .apply(lambda x: str(x) if isinstance(x, float) else x)
    .apply(lambda x: get_names(x,'display_name'))
    .apply(lambda x: " ".join(x) if isinstance(x,list) else x ) 
    .str.replace(r"[;,]", " ", regex=True)
    .str.replace(r"\.", "", regex=True)
    .str.replace(r"  ", " ", regex=True)
)

searchResults['pages'] = searchResults.iloc[:, 9].apply(_extract_range)

# rename columns in searchResults to match the other datasets
searchResults.columns = ['Title', 'source_title', 'Year', 'Month', 'Volume', 'Issue', 'Supplement', 'special_issue', 'article_number', 'pages',
'Authors', 'inventors', 'book_corp', 'book_editors', 'books', 'additional_authors', 'anonymous', 'assignees', 'Editors', 'record',
'references', 'related', 'doi', 'issn', 'eissn', 'isbn', 'eisbn','pmid', 'author_keywords', 'unique_type', 'uid']

searchResults['Year'] = searchResults['Year'].str.replace(r",", "", regex=True)

searchResults = searchResults.reset_index(drop=True)
searchResults.shape

# check for duplicated rows
searchResults[searchResults.duplicated(subset=['Title','Authors','Year'], keep=False)].sort_values(by=['Title','Authors','Year'])

# combine WoS search reaults to combined
combined = combined.map(str).map(str.upper).merge(
    searchResults.map(str).map(str.upper),
          on=['Title','Authors','Year'],suffixes =['_comb', '_wos'], validate='one_to_one', how="outer").reset_index(drop=True)

combined['Journal_Book_Institution_Meeting'] = [
    combined.loc[record,'source_title'] 
                if (pd.isna(combined.loc[record,'Journal_Book_Institution_Meeting']) or combined.loc[record,'Journal_Book_Institution_Meeting'] == None) 
                else combined.loc[record,'Journal_Book_Institution_Meeting'] for record in combined.index
                ]

combined = combined.loc[:, ['index', 'Title', 'Authors', 'Year', 'Index', 'Doc_Type',
       'Journal_Book_Institution_Meeting', 'Publishing_Details', 'Trace_Type',
       'Study_Type', 'Keywords', 'Abstract', 'Exp_Conditions_and_Results',
       'Relevance_to_Canada', 'Citation', 'Addressed question',
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
       'Raised questions (by authors)', 'Cautionary remarks', 
       'Month', 'Volume', 'Issue', 'Supplement', 'special_issue',
       'article_number', 'pages', 'inventors', 'book_corp', 'book_editors',
       'books', 'additional_authors', 'anonymous', 'assignees', 'Editors',
       'record', 'references', 'related', 'doi', 'issn', 'eissn', 'isbn',
       'eisbn', 'pmid', 'author_keywords', 'unique_type', 'uid']]

combined.shape
combined.columns
searchResults.columns

# reset format of all character strings to title case 
for col in combined.select_dtypes(include='object').columns:
    combined[col] = combined[col].str.title()

combined.to_csv('mergedDataSept.csv', encoding='utf-8', index=False)

combined = combined.replace({r"[nN]a[Nn]": pd.NA})
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

combined['allData'] = combined[cols].apply(_join_non_na, axis=1)

# The sentences to encode
dataAsList = combined['allData'].to_list()

#set domain specific stop words (ie forensic, bayesian etc)
custom_stopwords = ['forensic', 'bayesian', 'analysis','samples',
                    'evidence', 'examination', 'investigation','sample', 'examined'
                    'crime', 'method', 'testing', 'probabilistic', 'probabilitiy', 
                    'interpretation', 'likelihood', 'ratio','collected','analyze', 
                    'analyse', 'analysis', 'analyzed','extracted','specimens', 'spectrometry', 'extraction', 
                    'examiners','analyzing', 'findings', 'propositions', 'chromatography']

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