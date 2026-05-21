import pandas as pd
import os
import json
import re
import numpy as np
import util_functions as uf
import bibtexparser
from bibtexparser.bparser import BibTexParser
import datetime

# Load refs from John
with open('JBRefs.json', 'r', encoding='utf-8') as f:
    jb_refs = json.load(f)

jb_refs = pd.DataFrame.from_dict(jb_refs)

#Load zotero refs
parser = BibTexParser()
parser.ignore_nonstandard_types = False

with open('data/TPPR-total.bib', 'r', encoding='utf-8') as bibfile:
    bib_db = bibtexparser.load(bibfile, parser=parser)

zotero_refs = pd.DataFrame(bib_db.entries)
zotero_refs = zotero_refs.loc[:, ['title', 'date', 'author', 'journaltitle', 'keywords', 'publisher',
       'pages', 'abstract', 'doi', 'issn', 'volume','ENTRYTYPE', 'url','number',
       'type', 'institution','issue','isbn', 'edition']]

#Load combined DBs
modelledData = pd.read_csv('data/cleaned_modelReady_Apr.csv', encoding='utf-8')

#Prepare zotero refs for joining
zotero_refs['author'] = zotero_refs['author'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
zotero_refs['author'] = zotero_refs['author'].apply(
    lambda x: re.sub(r"\s*\.\s*", "", str(x)).upper().strip()
    )
zotero_refs['author'] = zotero_refs['author'].apply(lambda x: re.sub(r",*", "", str(x)))
zotero_refs = zotero_refs.reset_index(drop=True)

# fix date column to extract year only
zotero_refs['date'] = pd.to_datetime(zotero_refs['date'], format = 'mixed').dt.year

# Standardize column formatting for join keys
# Create normalized versions for joining
zotero_refs_norm = zotero_refs.copy()
zotero_refs_norm['title'] = zotero_refs_norm['title'].str.upper().str.strip()
zotero_refs_norm['date'] = zotero_refs_norm['date'].astype(str).str.strip()
zotero_refs_norm['journaltitle'] = zotero_refs_norm['journaltitle'].str.upper().str.strip()

# Encode string columns to bytes for joining to ensure special characters are handled correctly
zotero_refs_norm = zotero_refs_norm.apply(lambda x: x.str.encode('utf-8') if x.dtype == 'object' else x)

# remove unneeded columns from modelledData
modelledData = modelledData.drop(columns=['citing_articles', 'citations', 'corp',
 'investigators', 'sponsors', 'references', 'related', 'inventors', 'book_corp', 'books', 'anonymous',
 'assignees', 'record', 'additional_authors', 'Editors', 'article_number', 'Supplement', 'special_issue'])

# Standardize column formatting for join keys in modelledData
modelledData_norm = modelledData.copy()
modelledData_norm['Title'] = modelledData_norm['Title'].str.upper().str.strip()
modelledData_norm['Authors'] = modelledData_norm['Authors'].str.upper().str.strip()
modelledData_norm['Year'] = pd.to_numeric(modelledData_norm['Year'], errors='coerce').astype('Int64').astype(str).str.strip()
modelledData_norm['Journal_Book_Institution_Meeting'] = modelledData_norm['Journal_Book_Institution_Meeting'].str.upper().str.strip()

# Encode string columns to bytes for joining to ensure special characters are handled correctly
modelledData_norm = modelledData_norm.apply(lambda x: x.str.encode('utf-8') if x.dtype == 'object' else x)


combinedData = modelledData_norm.merge(
    zotero_refs_norm, 
    how='outer', 
    left_on=['Title', 'Authors', 'Year', 'Journal_Book_Institution_Meeting'], 
    right_on=['title', 'author', 'date', 'journaltitle'],
    suffixes=('_model', '_zotero')
)

# Merge back original columns from modelledData and zotero_refs where available, prioritizing modelledData values
combinedData = uf.fill_missing_values(
    combinedData,
    primary_cols=['Title', 'Authors', 'Year', 'Doc_Type', 'Journal_Book_Institution_Meeting', 'Abstract', 'doi_model','isbn_model','Keywords', 'Keywords'],
    alternate_cols=['title', 'author', 'date', 'ENTRYTYPE','journaltitle', 'abstract', 'doi_zotero', 'isbn_zotero','keywords', 'author_keywords']
)

retrieved_dois = pd.read_csv(
    'crossref_responses.csv',
    encoding='utf-8',
    usecols=[
        'title', 'issued', 'author', 'container-title', 'DOI', 'issue', 'page',
        'volume', 'publisher', 'type', 'alternative-id', 'ISSN', 'original-title'
    ]
)

retrieved_dois['issued'] = pd.to_numeric(retrieved_dois['issued'], errors='coerce').astype('Int64').astype(str).str.strip()
retrieved_dois['title'] = retrieved_dois['title'].str.upper().str.strip()
retrieved_dois['container-title'] = retrieved_dois['container-title'].str.upper().str.strip()

retrieved_dois_norm = retrieved_dois.apply(lambda x: x.str.encode('utf-8') if x.dtype == 'object' else x)

combinedData = combinedData.merge(
    retrieved_dois_norm, 
    how='outer', 
    left_on=['Title', 'Journal_Book_Institution_Meeting'], 
    right_on=['title', 'container-title'],
    suffixes=('_orig', '_retrieved')
)

# remove duplicated rows in combinedData
combinedData = combinedData.drop_duplicates(subset=['Title', 'Authors', 'Year', 'Journal_Book_Institution_Meeting'], keep='first')

combinedData = fill_missing_values(
    combinedData,
    primary_cols=['Title','doi_model', 'Issue', 'pages_model', 'Volume', 'publisher_orig', 'type_orig', 'issn_model'],
    alternate_cols=['title','DOI', 'issue_retrieved', 'page', 'volume_retrieved', 'publisher_retrieved', 'type_retrieved', 'ISSN']
)


# Standardize JB refs data
jb_refs['authors'] = jb_refs['authors'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
jb_refs['authors'] = jb_refs['authors'].apply(
    lambda x: re.sub(r"\s*\.\s*", "", str(x)).upper().strip()
    )
jb_refs['authors'] = jb_refs['authors'].apply(lambda x: re.sub(r",*", "", str(x)))
jb_refs = jb_refs.reset_index(drop=True)

# Standardize column formatting for join keys
# Create normalized versions for joining
jb_refs_norm = jb_refs.copy()
jb_refs_norm['title'] = jb_refs_norm['title'].str.upper().str.strip()
jb_refs_norm['year'] = jb_refs_norm['year'].astype(str).str.strip()
jb_refs_norm['container_title'] = jb_refs_norm['container_title'].str.upper().str.strip()

# Encode string columns to bytes for joining to ensure special characters are handled correctly
jb_refs_norm = jb_refs_norm.apply(lambda x: x.str.encode('utf-8') if x.dtype == 'object' else x)

# Perform outer join using normalized data 
combinedData = combinedData.merge(
    jb_refs_norm, 
    how='outer', 
    left_on=['Title', 'Authors', 'Year', 'Journal_Book_Institution_Meeting'], 
    right_on=['title', 'authors', 'year', 'container_title'],
    suffixes=('_modelledData', '_jbRefs')
)

# Decode byte columns back to strings for readability
combinedData = combinedData.apply(lambda x: x.str.decode('utf-8') if x.dtype == 'object' else x)


# Extract DOI from Publishing_Details or use existing DOI column
combinedData['doi'] = (combinedData['doi_model']
    .fillna(combinedData['DOI'])
    .fillna(
        combinedData['Publishing_Details']
        .str.extract(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', flags=re.IGNORECASE)[0]
    )
    .fillna(
        combinedData['Publishing_Details']
        .str.extract(r'(https?://[-._=?&;()/:A-Z0-9]+)(?=\s|$)', flags=re.IGNORECASE)[0]
    )
)

combinedData.drop(columns=['doi_model', 'title', 'author', 'container_title_jbRefs'], inplace=True)

# Export combined data
combinedData.to_csv('comparisonData.csv', index=False, encoding='utf-8')
# Export combined data
combinedData.to_csv('comparisonDataCheck.csv', index=False, encoding='utf-8')


import requests

def get_doi_by_title(title):
    # Queries Crossref for the title and returns the first result's DOI
    url = f"https://api.crossref.org/v1/works?query={title}&rows=1"
    response = requests.get(url).json()
    try:
        return response
    except:
        return pd.NA



response = [get_doi_by_title(row['Title']) for index, row in combinedData.iterrows() if pd.isna(row['doi'])]
pd.DataFrame(response).to_json('crossref_responses.json', orient='records', lines=True)
rawRefs = pd.read_json('crossref_responses.json', orient='records', lines=True)

# Returns a list of the index labels
indices = rawRefs['message'][rawRefs['message'].apply(lambda x: isinstance(x, list))].index.tolist()

rawRefs.loc[indices, 'message'] = pd.Series([x[0] for x in rawRefs.loc[indices,'message']], index=indices)
refItems = rawRefs['message'].apply(lambda x: x.get('items')[0] if len(x.get('items', [])) > 0 else pd.NA)
merged = {k: v for k, v in refItems.items() if pd.notna(v)}

dictRefs = pd.DataFrame(merged).T

dictRefs.to_csv('crossref_doi_results.csv', index=False, encoding='utf-8')

refs=pd.read_csv("data/bioRefs.csv", encoding='utf-8')
refs = refs.map(lambda s: s.title() if isinstance(s, str) else s)
refs['uid'] = [refs.loc[index, 'uid'] if not pd.isna(refs.loc[index, 'uid']) else refs.loc[index, 'pmid'] for index, row in refs.iterrows()]

# Compile acronym pattern once for efficiency
ACRONYM_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(a) for a in ["Dna", "Rna", "Str", "L'Adn", "Gsr", "Snp", "Pcr", "Lcn"]) + r')\b',
    re.IGNORECASE
)

# Convert acronyms to uppercase (match as whole words only)
refs = refs.map(lambda s: ACRONYM_PATTERN.sub(lambda m: m.group(0).upper(), s) if isinstance(s, str) else s)

refs['volume'] = refs['Publishing_Details'].str.extract(r'Vol\.?\s*(\d+)')
refs['number'] = refs['Publishing_Details'].str.extract(r'No\.?\s*(\d+)')
refs['pages'] = refs['Publishing_Details'].str.extract(r'P\.?:?\s*(\d+-?\d*)')

refs = refs.drop(columns=['Publishing_Details', 'eissn', 'isbn', 'eisbn', 'pmid'])


pattern = r'((?:[A-Z][a-z]{1,3}\s)?[A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s(\b[A-Za-z]{1,3}\b)(?=\s|$)'

def fix_and_join(text):
    if not isinstance(text, str): 
        return text
    
    # findall returns a list of (Name, Initials) tuples
    matches = re.findall(pattern, text)
    
    # Reconstruct with initials forced to UPPERCASE
    return ' and '.join([f"{initials.upper()} {name}" for name, initials in matches])


refs['Authors'] = refs['Authors'].apply(fix_and_join)

refs = refs[~refs['Trace_Type'].isin(['Fibres', 'Digital', 'Dental', 'Others', 'Hair; Others', 'Pathology', 'Anthropology', 'Trace', 'Documents', 'Bone', 'Firearms', 'Cosmetics', 'GSR', 'Shoeprint', 'Geotraces (Dust, Pollen, Soil)', 'Bloodstain', 'Fingermarks', 'Environmental'])]

refs.to_csv("data/bioRefs_cleaned.csv", index=False, encoding='utf-8')



combind = pd.read_csv('comparisonDataCheck.csv', encoding='utf-8').reset_index(drop=True)

combind =uf.fill_missing_values(combind, 
            ['Title','Title', 'Authors', 'Authors', 'Year','Journal_Book_Institution_Meeting','DOI', 'DOI', 'edition_modelledData','Volume',  'Volume', 'Issue', 'Issue','number','pages', 'pages', 'type', 'type', 'type','issn_zotero'], 
            ['title', 'original-title', 'authors', 'author', 'year','container_title', 'doi', 'doi_model','edition_jbRefs','volume_orig', 'volume', 'issue_orig', 'issue', 'article_number','pages_model', 'pages_zotero','types', 'type_orig','source_types','issn_model'])

combind['Year'] = pd.to_numeric(combind['Year'], errors='coerce').astype('Int64').astype(str).str.strip()
refs['Year'] = pd.to_numeric(refs['Year'], errors='coerce').astype('Int64').astype(str).str.strip()
cleanedRefs = refs.reset_index(drop=True)
newRefs = combind.merge(cleanedRefs, left_on=['Title', 'Authors', 'Year', 'Journal_Book_Institution_Meeting'], right_on=['Title', 'Authors','Year', 'Journal_Book_Institution_Meeting'], how='outer')

newRefs =uf.fill_missing_values(newRefs, 
            ['Doc_Type_x', 'Publishing_Details_x', 'Trace_Type_x', 'Study_Type_x', 'Keywords_x','Abstract_x', 'eissn_x', 'issn', 'isbn',
       'eisbn_x', 'pmid_x', 'uid_x', 'publisher',  'url_x', 'DOI'], 
            ['Doc_Type_y', 'Publishing_Details_y', 'Trace_Type_y', 'Study_Type_y', 'Keywords_y', 'Abstract_y', 'eissn_y', 'issn_zotero', 'isbn_model',
       'eisbn_y', 'pmid_y', 'uid_y', 'publisher_orig', 'url_y', 'doi'])

newRefs.columns = ['index', 'Title', 'Authors', 'Year', 'Doc_Type', 'Journal_Book_Institution_Meeting', 'Publishing_Details','Trace_Type', 'Study_Type', 'Keywords', 'Abstract',
                   'Exp_Conditions_and_Results', 'Relevance_to_Canada', 'Citation','Addressed question', 'Activity context', 'Category', 'Specifications','Variables of interest', 
                   'stringency of control', 'No of individuals','Replicates per Individual and condition', 'Nucleic Acid','Bodily origin', 'depositor characteristics',
                   'Criteria for shedder status', 'Previous activities', 'Contact scenario', 'Primary substrate type', 'Primary substrate Material', 'Deposit', 'Delay (conditions)', 
                   'Secondary substrate type', 'Secondary Substrate material','Type of secondary contact', 'Further transfer', 'Background DNA on sampled surface', 'Sampling time', 
                   'Persistance (conditions)', 'Sampling method', 'Sampling area','Extraction', 'DNA Quantification', 'Input for Profiling', 'Profiling','Reference samples', 
                   'Profile interpretation and mixture analysis','RNA data interpretation', 'DNA Quantitiy', 'Profile Quality','Parameter used for comparison', 'Summary of results',
                   'Raised questions (by authors)', 'Cautionary remarks', 'Month','Volume', 'Issue', 'issuing_organizations', 'eissn', 'eisbn','pmid', 'uid', 'url', 'number', 
                   'institution','edition', 'issued', 'container-title', 'alternative-id', 'raw', 'DOI', 'pages', 'type', 'editors', 'issn', 'isbn', 'publisher']
