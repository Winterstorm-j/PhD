
"""Utility helpers for the TPPRDB analysis project.
Import as:
    import util_functions
"""

import ast
import re
import pandas as pd
from typing import Any, List, Optional
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim import corpora

__all__ = [
    "get_names",
    "combine_group_rows",
    "preprocess_string_columns",
    "_extract_range",
    "preprocess",
]


def get_names(val: Any, col: str):
    """Parse a string representation of a list/dict and extract values for `col`."""
    if isinstance(val, str):
        try:
            data = ast.literal_eval(val)
            # If it's a list of dicts
            if isinstance(data, list):
                return [d[col] for d in data if isinstance(d, dict) and col in d]
            # If it's a single dict
            if isinstance(data, dict):
                return [data[col]] if col in data else []
            # fallback to raw string
            return val
        except (ValueError, SyntaxError):
            return val
    return val


def combine_group_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Combine rows in a DataFrame by grouping on specified columns and aggregating other columns.
    """
    def combine_series(series):
        unique_vals = series.dropna().unique()
        if len(unique_vals) == 0:
            return pd.NA
        if len(unique_vals) == 1:
            return unique_vals[0]
        return "; ".join([str(val) for val in unique_vals])

    grouped = df.groupby(group_cols, dropna=False)
    combined = grouped.agg(combine_series).reset_index()
    return combined


def preprocess_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase and strip only object/string columns (in-place) and return the DataFrame."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.upper()
            .str.strip()
            .replace(r'^(NAN|NA|NONE|NULL)$', pd.NA, regex=True)
        )
    return df


def _extract_range(x: Any) -> Optional[str]:
    """
    Extract a page/range string from a variety of inputs:
    - If input is a dict-like string, try to literal_eval and get 'range'
    - If it's a list/tuple, join with '-'
    - If it's a plain string, attempt to extract numeric ranges like "12-34" or "12 to 34"
    Returns None when nothing sensible can be extracted.
    """
    if pd.isna(x) or x in ('', 'nan', None):
        return None

    # If x is a string, try to interpret Python literal (e.g. "{'range': ['12','34']}")
    val = x
    if isinstance(x, str):
        s = x.strip()
        try:
            val = ast.literal_eval(s)
        except Exception:
            val = s

    # dict-like: get 'range' key
    if isinstance(val, dict):
        r = val.get('range')
        if r is None:
            return None
        if isinstance(r, (list, tuple)):
            parts = [str(i).strip() for i in r if i not in (None, '')]
            return "-".join(parts) if parts else None
        return str(r).strip() or None

    # list/tuple: join parts
    if isinstance(val, (list, tuple)):
        parts = [str(i).strip() for i in val if i not in (None, '')]
        return "-".join(parts) if parts else None

    # plain string: try to find numeric range
    s = str(val).strip()
    # match "12-34", "12 – 34", "12 to 34"
    m = re.search(r'(\d{1,5})\s*(?:[-–—]|to)\s*(\d{1,5})', s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # fallback: return the cleaned string if non-empty
    return s if s else None


def preprocess(text: Any, custom_stopwords: Optional[set] = None) -> str:
    """Minimal text preprocess: lowercase, keep words >=3 chars, remove custom stopwords."""
    if text is None or not isinstance(text, str):
        return ""
    if custom_stopwords is None:
        custom_stopwords = set()
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return ' '.join([w for w in words if w not in custom_stopwords])

def _join_non_na(row):
    vals = (
     row.dropna()
        .astype(str)
        .map(lambda s: s.strip())
    )
    vals = [v for v in vals if v and v.lower() not in ('nan', 'none', 'na')]
    return '; '.join(vals) if vals else pd.NA

# functions to evaluate topic model
# quantify topic coherence using normalised Pointwise Mutual Information coherence measure, 
# which tests pairwise agreement between every word in a topic
# and their co-occurrence in the original documents 
# (between -1 and 1, >0 is good, negative value indicate incoherent topics (less semantically similar))
# Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. 
# In Proceedings of the eighth ACM international conference on Web search and data mining (pp. 399–408).
# Gerlof Bouma. 2009. Normalized (pointwise) mutual information in collocation extraction. 
# Proceedings of Global Summit on Computing and Linguistics (GSCL), 30:31–40
def calculate_coherence_score(topic_model, docs):
    # Preprocess documents
    cleaned_docs = topic_model._preprocess_text(docs)

    # Extract vectorizer and tokenizer from BERTopic
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()
    # depending on the version and if you get an error use commented out code below:
    # words = vectorizer.get_feature_names()
    tokens = [tokenizer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # Create topic words
    topic_words = [[dictionary.token2id[w] for w in words if w in dictionary.token2id]
    for _ in range(len(topic_model.get_topics()))]

    # this creates a list of the token ids (in the format of integers) of the words in words that are also present in the 
    # dictionary created from the preprocessed text. The topic_words list contains list of token ids for each 
    # topic.

    coherence_model = CoherenceModel(topics=topic_words,
                                    texts=tokens,
                                    corpus=corpus,
                                    dictionary=dictionary,
                                    coherence='c_npmi')
    coherence = coherence_model.get_coherence()

    return coherence

# percentage of unique words across all topics (how distinct each topic is), between 0 and 1, higher is better.
# Theis is equivalent to the inverse of Jaccard Similarity, which is the intersect set of words over the union 
# set of words across all topics. Deng, F., Siersdorfer, S., Zerr, S.: Efficient jaccard-based diversity analysis of large
# document collections. In: Proceedings CIKM. (2012) 1402–1411
def calculate_diversity_score(topic_model):
    topics = topic_model.get_topics()
    unique_words = set()
    total_words = 0

    for topic_id, words_probs in topics.items():
        words = [word for word, prob in words_probs]
        unique_words.update(words)
        total_words += len(words)

    diversity_score = len(unique_words) / total_words if total_words > 0 else 0
    return diversity_score