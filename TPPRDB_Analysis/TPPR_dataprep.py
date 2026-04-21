#! /Users/jbuc045/Projects/.venv/bin/python
import os
# os.chdir('./PhD/TPPRDB_Analysis')

import pandas as pd
import numpy as np
import re
from util_functions import get_names, combine_group_rows, preprocess_string_columns, _extract_range

####### TTADB
# IMPORT TTADB removing any trailing whitespace
TPPR_db = pd.read_table("TTADB_Mar2026.txt", encoding='utf-8',header=None).map(str).map(str.strip).reset_index(drop=True)

# remove empty columns and rename last column
TPPR_db = TPPR_db.drop(TPPR_db.columns[8:11], axis=1)
TPPR_db.columns=['Study_Type','Keywords','Trace_Type','Exp_Conditions_and_Results','Relevance_to_Canada',
                 'Authors_Year_Title_Journal_Book_Institution_Meeting','Abstract','Doc_Type','Study_Type2','Trace_Type2','Citation','Index']

# split full citationin into Authors (words before the date, Year numbers between brackets, Title everything after) 
searchTerm = [re.search(r'(.+)\(([0-9sd\.]{4})\)(.*)',row) if row else () for row in TPPR_db['Authors_Year_Title_Journal_Book_Institution_Meeting']]
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
dnaTrac = pd.read_csv('data/DNA-TrAC_Ver-2019-12-16.csv', dtype=str, encoding='utf-8')

dnaTrac = dnaTrac.map(str).map(str.strip).reset_index(drop=True)

dnaTrac['Authors'] = (dnaTrac['Authors'].str.replace(r"[;,]", " ", regex=True)
                      .str.replace(r"\.", "", regex=True)
                      .str.replace(r"  ", " ", regex=True)
)

dnaTrac['Journal'] = dnaTrac['Journal'].str.replace(r"FSI[:]* ", "FORENSIC SCIENCE INTERNATIONAL: ", regex=True)
dnaTrac['Journal'] = dnaTrac['Journal'].str.replace("FSI", "FORENSIC SCIENCE INTERNATIONAL")
dnaTrac['Journal'] = dnaTrac['Journal'].str.replace("Science and Justice", "SCIENCE & JUSTICE")

newDnaTrac = combine_group_rows(dnaTrac, ['Title', 'Authors','Year'])

# find duplicated rows in newDnaTrac in title, authors, and year
duplicates = newDnaTrac[newDnaTrac.duplicated(subset=['Title', 'Authors','Year'], keep=False)]

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
    on=['Title','Authors','Year'],
    suffixes=['_dnatrac', '_ttadb'],
    validate='one_to_one',
    how="outer"
).reset_index()

combined.shape
combined['Journal_Book_Institution_Meeting'] = [
    combined.loc[record,'Journal'] 
                if (pd.isna(combined.loc[record,'Journal_Book_Institution_Meeting']) or 
                    combined.loc[record,'Journal_Book_Institution_Meeting'] == None) 
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
searchResults = pd.read_csv('data/articleListFull.csv', encoding='utf-8').map(str).map(str.strip).reset_index(drop=True)

#remove duplicates
searchResults = (
    searchResults.assign(_priority=searchResults['types'].str.contains("Book"))
      .sort_values("_priority", ascending=False)
      .drop_duplicates(subset='uid')
      .drop(columns="_priority")
)

# change results so all dictionary columns contain flat data
searchResults = searchResults.map(lambda x: np.nan if isinstance(x, str) and x == "nan" else x)

searchResults['authors'] = (
    searchResults['authors']
        .astype(str)
        .apply(lambda x: get_names(x, 'wos_standard'))
        .apply(lambda x: " ".join(x) if isinstance(x, list) else x)
        .str.replace(r"[;,]", " ", regex=True)
        .str.replace(r"\.", "", regex=True)
        .str.replace(r"  ", " ", regex=True)
)
searchResults['book_editors'] = (
    searchResults['book_editors']
        .astype(str)
        .apply(lambda x: get_names(x,'display_name'))
        .apply(lambda x: " ".join(x) if isinstance(x,list) else x )
        .str.replace(r"[;,]", " ", regex=True)
        .str.replace(r"\.", "", regex=True)
        .str.replace(r"  ", " ", regex=True)
)
searchResults['editors'] = (
    searchResults.loc[:,'editors']
    .astype(str)
    .apply(lambda x: get_names(x,'display_name'))
    .apply(lambda x: " ".join(x) if isinstance(x,list) else x ) 
)
searchResults['editors'] = ( #split so that mask can have up tp date Series in its condition
    searchResults['editors']
    .mask(searchResults['editors']=="nan", searchResults['book_editors'])
    .str.replace(r"[;,]", " ", regex=True)
    .str.replace(r"\.", "", regex=True)
    .str.replace(r"  ", " ", regex=True)
)

searchResults['pages'] = searchResults.iloc[:, 9].apply(_extract_range)

# rename columns in searchResults to match the other datasets
searchResults.columns = ['uid', 'Title', 'types', 'source_types', 'source_title', 'Year',
       'Month', 'Volume', 'Issue', 'Supplement', 'special_issue',
       'article_number', 'pages', 'Authors', 'inventors', 'book_corp',
       'book_editors', 'books', 'additional_authors', 'anonymous', 'assignees',
       'corp', 'Editors', 'investigators', 'sponsors', 'issuing_organizations',
       'record', 'citing_articles', 'references', 'related', 'citations',
       'doi', 'issn', 'eissn', 'isbn', 'eisbn', 'pmid', 'author_keywords']

searchResults['Year'] = searchResults['Year'].str.replace(r",", "", regex=True)

searchResults = searchResults.reset_index(drop=True)
searchResults.shape

# check for duplicated rows
searchResults[searchResults.duplicated(subset=['Title','Authors','Editors','Year'], keep=False)]\
.sort_values(by=['Title','Authors','Year', 'Editors'])

# combine WoS search reaults to combined
combined = combined.map(str).map(str.upper).merge(
    searchResults.map(str).map(str.upper),
          left_on=['Title','Authors','Journal_Book_Institution_Meeting','Year'],
          right_on=['Title','Authors','source_title','Year'],
          suffixes =['_comb', '_wos'], 
          validate='one_to_one', 
          how="outer").reset_index(drop=True)

combined['Journal_Book_Institution_Meeting'] = [
    combined.loc[record,'source_title'] 
                if (pd.isna(combined.loc[record,'Journal_Book_Institution_Meeting']) or 
                    combined.loc[record,'Journal_Book_Institution_Meeting'] == None) 
                else combined.loc[record,'Journal_Book_Institution_Meeting'] for record in combined.index
                ]

listOfCols = ["index", "Title", "Authors", "Year", "Index", "Doc_Type","Journal_Book_Institution_Meeting", "Publishing_Details", "Trace_Type","Study_Type", "Keywords", "Abstract", "Exp_Conditions_and_Results",
              "Relevance_to_Canada", "Citation", "Addressed question","Activity context", "Category", "Specifications","Variables of interest", "stringency of control", "No of individuals",
              "Replicates per Individual and condition", "Nucleic Acid","Bodily origin", "depositor characteristics","Criteria for shedder status", "Previous activities","Contact scenario", 
              "Primary substrate type","Primary substrate Material", "Deposit", "Delay (conditions)","Secondary substrate type", "Secondary Substrate material","Type of secondary contact", 
              "Further transfer","Background DNA on sampled surface", "Sampling time","Persistance (conditions)", "Sampling method", "Sampling area","Extraction", "DNA Quantification", "Input for Profiling", 
              "Profiling","Reference samples", "Profile interpretation and mixture analysis","RNA data interpretation", "DNA Quantitiy", "Profile Quality","Parameter used for comparison", "Summary of results",
              "Raised questions (by authors)", "Cautionary remarks", "Month", "Volume", "Issue", "Supplement", "special_issue","article_number", "pages", "inventors", "book_corp", "books", "additional_authors",
              "anonymous", "assignees", "Editors","record", "references", "related", "types","source_types","corp", "investigators", "sponsors","issuing_organizations", "citing_articles","citations","doi", "issn", 
              "eissn", "isbn","eisbn", "pmid", "author_keywords","uid"]


combined = combined.loc[:, listOfCols]



combined.shape
combined.columns
searchResults.columns

# reset format of all character strings to title case 
for col in combined.select_dtypes(include='object').columns:
    combined[col] = combined[col].str.title()



combined = combined.replace({r"[nN]a[Nn]": pd.NA})

combined.to_csv('data/mergedDataMar.csv', encoding='utf-8', index=False)
