
"""Utility helpers for the TPPRDB analysis project.
Import as:
    import util_functions
"""

import ast
import re
import math
import warnings
from copy import deepcopy
from typing import Any

import pandas as pd
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from bertopic import BERTopic
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics.pairwise import cosine_similarity
import umap
from lexicalrichness import LexicalRichness


__all__ = [
    'get_names',
    'combine_group_rows',
    'preprocess_string_columns',
    '_extract_range',
    'preprocess',
    '_join_non_na',
    'calculate_coherence_score',
    'calculate_diversity_score',
    'resolve_overlaps',
    'float_range',
    'find_and_replace',
    'fill_missing_values',
    'run_model',
    'cross_validate_zeroshot_bertopic',
]


def get_names(val: Any, col: str):
    """
    Parse a string representation of a list/dict and extract values for `col`.
    """
    
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


def combine_group_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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
    """
    Uppercase and strip only object/string columns (in-place) and return the DataFrame.
    """
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.upper()
            .str.strip()
            .replace(r'^(NAN|NA|NONE|NULL)$', pd.NA, regex=True)
        )
    return df


def _extract_range(x: Any) -> str|None:
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


def preprocess(text: Any, stopwords: list[Any]|None) -> str:
    """
    Minimal text preprocess: lowercase, keep words >=3 chars, remove custom stopwords.
    """
    
    if text is None or not isinstance(text, str):
        return ""
    if stopwords is None:
        custom_stopwords = set()
    else:
        custom_stopwords = set(stopwords)
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


def _prepare_vectorizer_corpus(docs: list[str], vectorizer_model):
    """
    Fit a vectorizer, tokenize documents, and return the dictionary and corpus.
    """
    vectorizer_model.fit(docs)
    tokenizer = vectorizer_model.build_tokenizer()
    tokens = [tokenizer(doc.lower()) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    return tokenizer, tokens, dictionary, corpus


def _get_topic_words(topic_model, exclude_outliers: bool = True) -> list[str]:
    """
    Extract topic words from a BERTopic model.
    """
    topic_info = topic_model.get_topic_info()
    if exclude_outliers and 'Topic' in topic_info.columns:
        topic_ids = topic_info[topic_info['Topic'] != -1]['Topic']
    else:
        topic_ids = topic_info['Topic']

    words: list[str] = []
    for topic_id in topic_ids:
        topic = topic_model.get_topic(topic_id)
        if topic:
            words.extend([word for word, _ in topic])
    return words


def calculate_topic_diversity_mtld(topic_model, min_words: int = 10) -> float:
    """
    Estimate topic diversity using MTLD over topic word tokens.
    """
    topic_words = _get_topic_words(topic_model)
    if len(topic_words) <= min_words:
        return 0.0
    try:
        lex = LexicalRichness(' '.join(topic_words))
        return float(lex.mtld())
    except Exception:
        return 0.0


def _compute_silhouette_score(embeddings, topics) -> float:
    """
    Compute silhouette score for clustered documents only.
    """
    valid_idx = [i for i, t in enumerate(topics) if t != -1]
    if len(valid_idx) <= 2:
        return -1.0

    valid_topics = np.array(topics)[valid_idx]
    if len(np.unique(valid_topics)) < 2:
        return -1.0

    try:
        sil_score = silhouette_score(np.asarray(embeddings)[valid_idx], valid_topics)
    except Exception:
        return -1.0
    return float(sil_score) if not np.isnan(sil_score) else -1.0


def _create_bertopic_model(
    vectorizer_model,
    embedding_model=None,
    candidate_topics=None,
    nr_topics=None,
    min_topic_size=2,
    random_state=38,
    calculate_probabilities=False,
):
    """
    Instantiate a BERTopic model with stable zero-shot and UMAP settings.
    """
    return BERTopic(
        vectorizer_model=vectorizer_model,
        umap_model=umap.UMAP(random_state=random_state),
        zeroshot_topic_list=candidate_topics,
        zeroshot_min_similarity=0.8,
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=calculate_probabilities,
        nr_topics=nr_topics,
    )


def _outlier_ratio(topics) -> float:
    """
    Compute the outlier ratio for BERTopic output.
    """
    topics_str = [str(t) for t in topics]
    outliers = topics_str.count("-1") + topics_str.count("-1.0") + topics_str.count("unknown")
    return outliers / len(topics_str) if topics_str else 0.0


def _count_valid_topics(topics) -> int:
    """
    Count valid topics excluding outliers.
    """
    topics_str = set(str(t) for t in topics)
    return len(topics_str - {"-1", "-1.0", "unknown"})

# functions to evaluate topic model

def calculate_coherence_score(topic_model, dictionary=None, corpus=None, tokens=None):
    ''' 
    Quantifies topic coherence using normalised Pointwise Mutual Information coherence measure, which tests 
    pairwise agreement between every word in a topic and their co-occurrence in the original documents
    (between -1 and 1, >0 is good, negative value indicate incoherent topics (less semantically similar))
    - Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. 
      In Proceedings of the eighth ACM international conference on Web search and data mining (pp. 399-408).
    - Gerlof Bouma. 2009. Normalized (pointwise) mutual information in collocation extraction. 
      Proceedings of Global Summit on Computing and Linguistics (GSCL), 30:31-40
    '''
    if dictionary is None or tokens is None:
        return 0.0
    if not tokens or len(tokens) == 0:
        return 0.0
      
    # create a list of the token ids (as integers) of the words in the vector called words that 
    # are also present in the dictionary created from the preprocessed text. Deals with multiword tokens
    # Iterates through each topic ID and its list of (word, score) tuples
    topic_words = []
    
    # Loop through all topics (both zero-shot string keys and numerical sub-clusters)
    for topic_id in topic_model.get_topics():
        # Skip standard outlier flag
        if topic_id in [-1, "-1", "-1.0", "unknown"]:
            continue
            
        extracted_tokens = []
        topic_data = topic_model.get_topic(topic_id)
        
        if topic_data:
            for word_entry, score in topic_data:
                # 1. Clean & split the word entry in case it's a multi-word phrase or n-gram
                individual_words = word_entry.lower().split()
                
                for word in individual_words:
                    # 2. Match against your Gensim text dictionary
                    if word in dictionary.token2id:
                        extracted_tokens.append(dictionary.token2id[word])
                        
        if extracted_tokens:
            topic_words.append(extracted_tokens)
    
    # Avoid invoking Gensim coherence on empty or degenerate topic sets.
    if not topic_words or len(topic_words) < 2:
        return 0.0
      
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            coherence_model = CoherenceModel(topics=topic_words,
                                            texts=tokens,
                                            corpus=corpus,
                                            dictionary=dictionary,
                                            coherence='c_npmi')
            coherence = coherence_model.get_coherence()
        except Exception:
            return 0.0
    
    # Checks if coherence is NOT nana or close to 0
    if isinstance(coherence, (list, tuple, np.ndarray)):
        coherence = np.nanmean(coherence) if len(coherence) else 0.0
    if np.isnan(coherence) or math.isclose(coherence, 0, abs_tol=1e-9):
        coherence = 0.0

    return float(coherence)


def calculate_diversity_score(topic_model):
    ''' 
    Scores the model based on percentage of unique words across all topics (how distinct each topic is), between 0 and 1, 
    higher is better. This is equivalent to the inverse of Jaccard Similarity, which is the intersect set of words over 
    the union set of words across all topics. 
    Deng, F., Siersdorfer, S., Zerr, S.: Efficient jaccard-based diversity analysis of large document collections. 
    In: Proceedings CIKM. (2012) 1402-1411
    '''
    topics = topic_model.get_topics()
    unique_words = set()
    total_words = 0

    for topic_id, words_probs in topics.items():
        words = [word for word, prob in words_probs]
        unique_words.update(words)
        total_words += len(words)

    diversity_score = len(unique_words) / np.sqrt(total_words) if total_words > 0 else 0
    return diversity_score


def resolve_overlaps(positions, markers, min_dist, steps=40, repulsion_strength=1.0,
        marker_repulsion_strength=2.0, attraction_strength=0.05, max_step=0.5):
    '''
    Resolve overlaps using a force-directed (physics simulation) method.
    Fully vectorized with NumPy. Very fast.

    positions: (N, 2) initial label positions
    markers: (N, 2) anchor points
    min_dist: minimum allowed distance between labels
    steps: number of physics iterations (20-40 usually enough)
    '''

    N = len(positions)
    pos = positions.astype(float)

    for _ in range(steps):

        ## Repulsion between labels
        diff = pos[:, None, :] - pos[None, :, :]        # (N, N, 2)
        dist = np.linalg.norm(diff, axis=2) + 1e-12     # avoid div/0

        # Mask: only labels within min_dist, ignore self (dist=0)
        mask = (dist < min_dist) & (dist > 0)

        # Repulsion magnitude: linear penalty
        overlap_amount = (min_dist - dist) * mask

        # Normalize direction vectors
        direction = diff / dist[..., None]

        # Force = direction × overlap
        repel_forces = (direction * overlap_amount[..., None]).sum(axis=1)

        # Scale repulsion strength
        repel_forces *= repulsion_strength

        ## Repulsion from markers
        m_diff = pos - markers
        m_dist = np.linalg.norm(m_diff, axis=1) + 1e-12

        too_close = m_dist < min_dist
        m_overlap = (min_dist - m_dist) * too_close
        m_direction = m_diff / m_dist[:, None]

        marker_forces = m_direction * m_overlap[:, None] * marker_repulsion_strength

        ## Attraction toward markers
        # pulls labels back gently so they don't drift endlessly
        attraction = (markers - pos) * attraction_strength


        ## Total force & update positions
        total_force = repel_forces + marker_forces + attraction

        # Limit movement per step for stability
        step_move = np.clip(total_force, -max_step, max_step)

        pos += step_move

    return pos

def float_range(start, stop, step):
    """
    Create a range generator for a float (normal range function is integer only)
    """
    
    current = start
    while current < stop:
        yield current
        current += step
        
# Apply the logic only where Trace_Type is empty, with a default value of 'NA' for no matches
def find_and_replace(df, findColumn, replaceColumn, conditions, choices):
    if replaceColumn not in df.columns:
        df[replaceColumn] = ''
    
    mask = df[replaceColumn] == ''
    df.loc[mask, replaceColumn] = np.select(conditions, choices, default='NA')
    
    
def fill_missing_values(df, primary_cols, alternate_cols):
    """
    Fill missing values in primary columns from alternate columns.
    
    Args:
        df: DataFrame to modify
        primary_cols: List of primary column names
        alternate_cols: List of alternate column names (same order as primary_cols)
    
    Returns:
        Modified DataFrame
    """
    for primary, alternate in zip(primary_cols, alternate_cols):
        df[primary] = np.where(
            df[primary].isna() & ~df[alternate].isna(), 
            df[alternate], 
            df[primary]
        )
    
        df.drop(columns=alternate, inplace=True)
    return df

def run_model(docs, embeddings, vectorizer_model, notopics, clustermodel=None, embedding_model=None, candidate_topics=None):
    '''
    Train a BERTopic model and return fitted metrics plus the model.
    '''
    vectorizer = deepcopy(vectorizer_model)
    topic_model = _create_bertopic_model(
        vectorizer_model=vectorizer,
        embedding_model=embedding_model,
        candidate_topics=candidate_topics,
        nr_topics=notopics,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    _, tokens, dictionary, corpus = _prepare_vectorizer_corpus(docs, vectorizer)

    coherence = calculate_coherence_score(topic_model, dictionary=dictionary, corpus=corpus, tokens=tokens)
    diversity = calculate_topic_diversity_mtld(topic_model)
    sil_score = _compute_silhouette_score(embeddings, topics)
    score = coherence * 0.7 + diversity * 0.15 + sil_score * 0.15

    results = [(notopics, coherence, diversity, sil_score, score)]
    return [results, topic_model]

def cross_validate_zeroshot_bertopic(
    docs: list[str], 
    embeddings: np.ndarray = None,
    vectorizer_model=None,
    embedding_model: object = None,
    topic_list: list[str] = None,
) -> dict[str, Any]:
    """
    Perform repeated cross-validation for a zero-shot BERTopic model.
    """
    folds = RepeatedKFold(n_splits=5, n_repeats=5)
    fold_outlier_ratios: list[float] = []
    fold_topic_counts: list[int] = []
    all_fold_vocab_vectors: list[np.ndarray] = []

    global_vectorizer = deepcopy(vectorizer_model)
    global_vectorizer.fit(docs)
    vocabulary_size = len(global_vectorizer.vocabulary_)

    data_arr = np.array(docs)
    embeddings_arr = np.array(embeddings)

    for train_idx, _ in folds.split(data_arr):
        train_text = data_arr[train_idx].tolist()
        train_embeddings = embeddings_arr[train_idx]
        calculated_min_size = max(2, min(5, len(train_text) // 5))

        local_vectorizer = deepcopy(global_vectorizer)
        topic_model = _create_bertopic_model(
            vectorizer_model=local_vectorizer,
            embedding_model=embedding_model,
            candidate_topics=topic_list,
            min_topic_size=calculated_min_size,
            nr_topics=29,
            calculate_probabilities=False,
        )

        topics, _ = topic_model.fit_transform(train_text, embeddings=train_embeddings)
        fold_outlier_ratios.append(_outlier_ratio(topics))
        fold_topic_counts.append(_count_valid_topics(topics))

        if hasattr(topic_model, "c_tf_idf_") and topic_model.c_tf_idf_ is not None and topic_model.c_tf_idf_.shape[0] > 0:
            vocab_vector = topic_model.c_tf_idf_.sum(axis=0).A1
            if len(vocab_vector) < vocabulary_size:
                padded_vocab = np.zeros(vocabulary_size)
                padded_vocab[: len(vocab_vector)] = vocab_vector
                vocab_vector = padded_vocab
            elif len(vocab_vector) > vocabulary_size:
                vocab_vector = vocab_vector[:vocabulary_size]
        else:
            vocab_vector = np.zeros(vocabulary_size)

        if np.isnan(vocab_vector).any():
            vocab_vector = np.zeros(vocabulary_size)

        all_fold_vocab_vectors.append(vocab_vector)

        _, tokens, dictionary, corpus = _prepare_vectorizer_corpus(train_text, local_vectorizer)
        coherence_score = calculate_coherence_score(topic_model, dictionary=dictionary, corpus=corpus, tokens=tokens)
        diversity_score = calculate_topic_diversity_mtld(topic_model)
        silhouette_score = _compute_silhouette_score(train_embeddings, topics)
        overall_score = coherence_score * 0.7 + diversity_score * 0.15 + silhouette_score * 0.15
        print(f"Fold metrics: Coherence={coherence_score:.4f}, Diversity={diversity_score:.4f}, Silhouette={silhouette_score:.4f}, Overall={overall_score:.4f}")

    mean_outliers = float(np.mean(fold_outlier_ratios)) if fold_outlier_ratios else 0.0
    std_outliers = float(np.std(fold_outlier_ratios)) if fold_outlier_ratios else 0.0
    mean_topics = float(np.mean(fold_topic_counts)) if fold_topic_counts else 0.0

    max_len = max((len(v) for v in all_fold_vocab_vectors), default=0)
    padded_vectors = [np.pad(np.asarray(v), (0, max_len - len(v)), 'constant', constant_values=0)
                      for v in all_fold_vocab_vectors]
    uniform_matrix = np.array(padded_vectors)

    if uniform_matrix.size == 0 or len(uniform_matrix) < 2:
        mean_stability = 0.0
    else:
        similarity_matrix = cosine_similarity(uniform_matrix)
        np.fill_diagonal(similarity_matrix, np.nan)
        mean_stability = float(np.nanmean(similarity_matrix)) if not np.all(np.isnan(similarity_matrix)) else 0.0

    return {
        "mean_outlier_ratio": mean_outliers,
        "std_outlier_ratio": std_outliers,
        "mean_topic_count": mean_topics,
        "topic_structural_stability": float(mean_stability),
        "raw_metrics_per_fold": {
            "outlier_ratios": fold_outlier_ratios,
            "topic_counts": fold_topic_counts,
            
        },
    }
    
