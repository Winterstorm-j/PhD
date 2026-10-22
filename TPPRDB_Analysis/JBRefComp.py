import pandas as pd
import os

os.chdir('TPPRDB_Analysis')

import json
import ast
import re
import numpy as np

import bibtexparser 
import bibtexparser.middlewares as m
from bibtexparser.model import Entry, Field
import datetime
from namematcher import NameMatcher as nm
import util_functions as uf
from importlib import reload

reload(uf)

#  lambda logic to fill empty cells of a grouped DataFrame 
row_combiner = lambda x: ( pd.DataFrame(x).T if isinstance(x, pd.Series) else x).ffill().bfill().iloc[0]

nm_engine = nm()

#Load zotero refs
bib_db = bibtexparser.parse_file('./data/TPPR.bib', append_middleware=[m.SeparateCoAuthors()])

#Load combined DBs
modelledData = pd.read_csv('./data/cleaned_modelReady_Oct2026.csv', encoding='utf-8')

# remove unneeded columns from modelledData
modelledData = modelledData.drop(columns=['index','citing_articles', 'citations', 'corp',
 'investigators', 'sponsors', 'references', 'related', 'inventors', 'book_corp', 'books', 'anonymous',
 'assignees', 'record', 'additional_authors', 'Editors', 'article_number', 'Supplement', 'special_issue'])

modelledData = modelledData.fillna(pd.NA)
modelledData = modelledData.replace(r'^\s*$', pd.NA, regex=True)
modelledData = modelledData.replace([None, 'None', 'nan', 'NaN','<NA>'], pd.NA)
modelledData = modelledData.map(lambda x: pd.NA if isinstance(x, str) and x.strip() == '' else x)
modelledData = modelledData.map(lambda x: pd.NA if isinstance(x, float) and np.isnan(x) else x)

# Standardize column formatting for join keys in modelledData
modelledData_norm = modelledData.copy()
modelledData_norm['Title'] = modelledData_norm['Title'].apply(lambda x: re.sub(r"[\{\}']", "", str(x)).upper().strip() if pd.notna(x) else x)

modelledData_norm['Authors'] = modelledData_norm['Authors'].apply(uf.separate_author_initials)
modelledData_norm['Authors'] = modelledData_norm['Authors'].str.upper()
modelledData_norm['Authors_dict'] = [nm_engine.parse_names_list(entry) if pd.notna(entry) else None for entry in modelledData_norm['Authors']]

# remove trailing commas and replace semicolons with commas in the 'Authors' column
modelledData_norm['Authors_str'] = modelledData_norm['Authors'].apply(lambda x: re.sub(r"\,\s*$", "", x) if pd.notna(x) else x)
modelledData_norm['Authors_str'] = modelledData_norm['Authors_str'].apply(lambda x: re.sub(r"\b(\w)\s(\w)\b", r"\1\2", x))
modelledData_norm['Authors_str'] = modelledData_norm['Authors_str'].str.replace(r';', ',', regex=True)


modelledData_norm['Authors_str'] =  modelledData_norm['Authors_str'].apply(lambda x: x.upper().strip())
modelledData_norm['Year'] = pd.to_numeric(modelledData_norm['Year'], errors='coerce').astype('Int64').astype(str).str.strip()
modelledData_norm['Journal_Book_Institution_Meeting'] = modelledData_norm['Journal_Book_Institution_Meeting'].apply(lambda x: re.sub(r"[\{\}']", "", str(x)).upper().strip() if pd.notna(x) else x)

model_authors = [
    author for author_list in modelledData_norm['Authors_dict']
    if isinstance(author_list, list) for author in author_list
]

entry_list = []
for entry in bib_db.entries:
    # Blend core attributes (key, entry_type) with the rest of the fields
    entry_data = uf.lib_to_dict(entry)
    entry_list.append(entry_data)


zotero_refs = pd.DataFrame(entry_list)
zotero_refs=zotero_refs.apply(lambda x: uf.convert_fields(x) if x.apply(lambda y: isinstance(y, bibtexparser.model.Field)).any() else x)

zotero_refs = zotero_refs.loc[:, ['title', 'date', 'author', 'journaltitle', 'keywords', 'publisher',
       'pages', 'abstract', 'doi', 'issn', 'volume','ENTRYTYPE', 'url','number',
       'type', 'institution','issue','isbn', 'edition']]

zotero_refs['abstract'] = zotero_refs['abstract'].str.replace(r'\n', ' ', regex=True)

#Prepare zotero refs for joining
zotero_refs['Authors_str'] = zotero_refs['author'].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)
zotero_refs['Authors_str'] = zotero_refs['Authors_str'].str.upper()
zotero_refs['Authors_dict'] = [[nm_engine.parse_name(author) for author in entry.split('; ')] if pd.notna(entry) else None for entry in zotero_refs['Authors_str']]

# Add to masterlist of authors for matching
model_authors.extend([author for author_list in zotero_refs['Authors_dict']
    if isinstance(author_list, list) for author in author_list
])

#remove duplicate authors from model_authors
model_authors = uf.remove_duplicate_dicts(model_authors)

zotero_refs['Authors_str'] = zotero_refs['Authors_str'].apply(lambda x: re.sub(r"[\[\]'\{\}\.\,]", "", str(x)).upper().strip())
zotero_refs['Authors_str'] = zotero_refs['Authors_str'].apply(lambda x: re.sub(r"\b(\w)\s(\w)\b", r"\1\2", x))
zotero_refs['Authors_str'] = zotero_refs['Authors_str'].str.replace(r';', ',', regex=True)
# zotero_refs = zotero_refs.reset_index(drop=True)

zotero_refs['date'] = pd.to_datetime(zotero_refs['date'], format='mixed').dt.year.astype('Int64').astype(str).str.strip()

# standardise all empty strings, NaNs, None and <NA> values in the zotero_refs Dataframe to be true NAs
zotero_refs = zotero_refs.replace(r'^\s*$', pd.NA, regex=True)
zotero_refs = zotero_refs.replace([None, 'None', 'nan', 'NaN', 'NAN', '[nan nan]','<NA>'], pd.NA)
zotero_refs = zotero_refs.map(lambda x: pd.NA if isinstance(x, str) and x.strip() == '' else x)
zotero_refs = zotero_refs.map(lambda x: pd.NA if isinstance(x, float) and np.isnan(x) else x)
# zotero_refs = zotero_refs.map(lambda x: pd.NA if isinstance(x, (list, dict)) and not x else x)
zotero_refs = zotero_refs.fillna(pd.NA)


# Standardize column formatting for join keys
# Create normalized versions for joining
zotero_refs_norm = zotero_refs.copy()
zotero_refs_norm['title'] = zotero_refs_norm['title'].apply(lambda x: re.sub(r"[\{\}']", "", str(x)).upper().strip() if pd.notna(x) else x)
zotero_refs_norm['journaltitle'] = zotero_refs_norm['journaltitle'].apply(lambda x: re.sub(r"[\{\}']", "", str(x)).upper().strip() if pd.notna(x) else x)

# Encode string columns to bytes for joining to ensure special characters are handled correctly
zotero_refs_norm = zotero_refs_norm.apply(lambda x: x.str.encode('utf-8') if x.dtype == 'str' else x)

modelledData_norm = modelledData_norm.apply(lambda x: x.str.encode('utf-8') if x.dtype == 'str' else x)


combinedData = modelledData_norm.merge(
    zotero_refs_norm, 
    how='outer', 
    left_on=['Title', 'Year', 'Journal_Book_Institution_Meeting','Authors_str','doi'], 
    right_on=['title','date', 'journaltitle','Authors_str','doi'],
    suffixes=('', '_zotero'),
    validate='1:1'
)


# Merge back original columns from modelledData and zotero_refs where available, prioritizing modelledData values
combinedData = uf.fill_missing_values(
    combinedData,
    primary_cols=['Title', 'Year', 'Doc_Type', 'Journal_Book_Institution_Meeting', 'Abstract','Volume','Issue',
                  'pages','types','isbn','issn','Keywords', 'Keywords', 'Authors_dict'],
    alternate_cols=['title', 'date', 'ENTRYTYPE','journaltitle', 'abstract', 'volume','issue',
                    'pages_zotero','type','isbn_zotero','issn_zotero','keywords', 'author_keywords', 'Authors_dict_zotero']
)



# Run the row combiner and build the final DataFrame directly
collected_rows = uf.build_nested_groups(combinedData, ['doi', 'Authors_str','Year', 'Journal_Book_Institution_Meeting'], row_combiner)
combinedData = pd.DataFrame(collected_rows).reset_index(drop=True)


# Load refs from John
with open('data/TPPRsearchResults.json', 'r', encoding='utf-8') as f:
    jb_refs = json.load(f)

jb_refs_raw = jb_refs['data']
jbrefs_list = []

for ref in jb_refs_raw:
    ref_data = pd.DataFrame.from_dict(ref, orient='index').T
    jbrefs_list.append(ref_data)
    
jb_refs = pd.concat(jbrefs_list, ignore_index=True)

jb_refs['doi'] = [ref.get('DOI') if isinstance(ref, dict) else pd.NA for ref in jb_refs['externalIds']]

jb_refs.drop_duplicates(subset=['doi'], keep='first', inplace=True)

# Standardize JB refs data
jb_refs['Authors_str'] = jb_refs['authors'].apply(lambda x: ', '.join(map(str, [author.get('name') for author in x])) if isinstance(x, list) else x)
jb_refs['Authors_str'] = jb_refs['Authors_str'].apply(
    lambda x: re.sub(r"\s*\.\s*", " ", str(x)).upper().strip()
    )

jb_refs['Authors_dict'] = [[nm_engine.parse_name(author) for author in entry.split(', ')] if pd.notna(entry) else None for entry in jb_refs['Authors_str']]

jb_refs['Authors_str'] = jb_refs['Authors_dict'].apply(lambda x: ', '.join([f"{author.get('last_name', '')} {author.get('first_names', '')}" for author in x]) if isinstance(x, list) else x)
jb_refs['Authors_str'] = jb_refs['Authors_str'].apply(lambda x: re.sub(r"[\[\]'\.]", "", str(x)).upper().strip())
jb_refs['Authors_str']= jb_refs['Authors_str'].apply(lambda x: re.sub(r"\,\s*$", "", x))
jb_refs['Authors_str'] = jb_refs['Authors_str'].str.replace(r';', ',', regex=True)

# standardise all empty strings, NaNs, None and <NA> values in the zotero_refs Dataframe to be true NAs
jb_refs = jb_refs.replace(r'^\s*$', pd.NA, regex=True)
jb_refs = jb_refs.replace([None, 'None', 'nan', 'NaN', 'NAN', '[nan nan]','<NA>'], pd.NA)
jb_refs = jb_refs.map(lambda x: pd.NA if isinstance(x, str) and x.strip() == '' else x)
jb_refs = jb_refs.map(lambda x: pd.NA if isinstance(x, float) and np.isnan(x) else x)
jb_refs = jb_refs.fillna(pd.NA)

jb_refs = jb_refs.reset_index(drop=True)

# Standardize column formatting for join keys
# Create normalized versions for joining
jb_refs_norm = jb_refs.copy()
jb_refs_norm['title'] = jb_refs_norm['title'].str.upper().str.strip()
jb_refs_norm['year'] = jb_refs_norm['year'].astype(str).str.strip()
jb_refs_norm['journal'] = [journal.get('name') if isinstance(journal, dict) else journal for journal in jb_refs_norm['journal']]
jb_refs_norm['journal'] = jb_refs_norm['journal'].str.upper().str.strip()

# Encode string columns to bytes for joining to ensure special characters are handled correctly
jb_refs_norm = jb_refs_norm.apply(lambda x: x.astype(str).str.encode('utf-8') if x.dtype == 'str' else x)

# Perform outer join using normalized data 
combinedData = combinedData.merge(
    jb_refs_norm, 
    how='outer', 
    left_on=['doi','Title', 'Authors_str', 'Year', 'Journal_Book_Institution_Meeting'], 
    right_on=['doi','title', 'Authors_str', 'year', 'journal'],
    suffixes=('', '_jbRefs'),
    validate='1:1'
)


combinedData = uf.fill_missing_values(
    combinedData,
    primary_cols=[ 'url', 'Title', 'Year', 'Journal_Book_Institution_Meeting', 'Abstract', 'Authors_dict'],
    alternate_cols=[ 'url_jbRefs','title', 'year', 'journal', 'abstract', 'Authors_dict_jbRefs']
)


collected_rows = uf.build_nested_groups(combinedData, ['doi', 'Authors_str','Year', 'Journal_Book_Institution_Meeting'], row_combiner)
combinedData = pd.DataFrame(collected_rows).reset_index(drop=True)

retrieved_dois = pd.read_csv(
    'data/crossref_responses.csv',
    encoding='utf-8',
    usecols=[
        'title', 'issued', 'author', 'container-title', 'DOI', 'issue', 'page',
        'volume', 'publisher', 'type', 'alternative-id', 'ISSN', 'original-title'
    ]
)

retrieved_dois['issued'] = pd.to_numeric(retrieved_dois['issued'], errors='coerce').astype('Int64').astype(str).str.strip()
retrieved_dois['title'] = retrieved_dois['title'].str.upper().str.strip()
retrieved_dois['container-title'] = retrieved_dois['container-title'].str.upper().str.strip()

# standardise the author representation in the retrieved_dois DataFrame to match the modelledData and zotero_refs DataFrames
retrieved_dois['author'] = retrieved_dois['author'].astype(object)
retrieved_dois.loc[:,'author'] = retrieved_dois.loc[:,'author'].apply(lambda d: uf.rename_key(d, {'given': 'first_names','family': 'last_name'}))

retrieved_dois.drop_duplicates(subset=['DOI'], keep='first', inplace=True)

retrieved_dois['Authors_str'] =  retrieved_dois['author'] .apply(lambda x: ', '.join([f"{author.get('last_name', '')} {author.get('first_names', '')}" for author in x]) if isinstance(x, list) else x)
retrieved_dois['Authors_str'] = retrieved_dois['Authors_str'].apply(lambda x: re.sub(r"^\s*,\s*|\.", "", str(x)).upper().strip() if isinstance(x, str) else pd.NA)
retrieved_dois['Authors_str']= retrieved_dois['Authors_str'].apply(lambda x: re.sub(r"\,\s*$", "", x) if isinstance(x,str) else pd.NA)
retrieved_dois['Authors_str'] = retrieved_dois['Authors_str'].str.replace(r';', ',', regex=True)
retrieved_dois_norm = retrieved_dois.apply(lambda x: x.str.encode('utf-8') if x.dtype == 'str' else x)

combinedData = combinedData.merge(
    retrieved_dois_norm, 
    how='outer', 
    left_on=['doi','Title', 'Authors_str','Journal_Book_Institution_Meeting', 'Year'], 
    right_on=['DOI','title', 'Authors_str','container-title', 'issued'],
    suffixes=('', '_retrieved'),
    validate='1:1'
)

combinedData = uf.fill_missing_values(
    combinedData,
    primary_cols=['Title','doi', 'Issue', 'pages', 'Volume', 'publisher', 'types', 'issn', 'Journal_Book_Institution_Meeting', 'Authors_dict'],
    alternate_cols=['title','DOI', 'issue', 'page', 'volume', 'publisher_retrieved', 'type', 'ISSN', 'container-title', 'author_retrieved']
)

collected_rows = uf.build_nested_groups(combinedData, ['doi', 'Authors_str','Year', 'Journal_Book_Institution_Meeting'], row_combiner)
combinedData = pd.DataFrame(collected_rows).reset_index(drop=True)

combinedData['Authors'] = combinedData['Authors_str']

combinedData = uf.fill_missing_values(combinedData, primary_cols=['Doc_Type', 'Year', 'Year', 'Doc_Type'], alternate_cols=['types', 'issued', 'publicationDate','publicationTypes'])


# Decode byte columns back to strings for readability
combinedData = combinedData.apply(lambda x: x.str.decode('utf-8') if isinstance(x, bytes) else x)

# combinedData.drop(columns=['doi_model', 'title', 'author', 'container_title_jbRefs'], inplace=True)
# combinedData = combinedData.apply(lambda x: x.str.replace(r'\t', ' ', regex=True) if x.dtype == 'object' else x)
# Export combined data
combinedData.to_csv('comparisonData_1026.csv', index=False, encoding='utf-8', sep='\t')
# Export combined data
combinedData.to_csv('comparisonDataCheck.csv', index=False, encoding='utf-8', sep='\t')


import requests

def get_doi_by_title(title):
    # Queries Crossref for the title and returns the first result's DOI
    url = f"https://api.crossref.org/v1/works?query={title}&rows=1"
    response = requests.get(url).json()
    try:
        return response
    except:
        return pd.NA



response = [get_doi_by_title(row['Title']) for index, row in combinedData.iterrows() if pd.isna(row['doi_modelledData'])]
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



combind = pd.read_csv('comparisonData_0826.csv', encoding='utf-8',  sep='\t').reset_index(drop=True)

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
