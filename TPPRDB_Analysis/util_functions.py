
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
    Extract values for a key from a dictionary or a string representation of one.

    Args:
        val: A dictionary/list, a string containing a Python literal representation,
            or another value to pass through unchanged.
        col: Dictionary key to extract.

    Returns:
        A list of matching values for dictionary/list inputs; the original string
        when parsing fails or yields another type; otherwise the original value.
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
    Group rows and combine each non-key column's distinct non-null values.

    A column with no non-null values becomes ``pd.NA``; a single distinct value
    is retained as-is; multiple values are joined with ``"; "``.

    Args:
        df: DataFrame to group.
        group_cols: Columns that identify each group.

    Returns:
        A new DataFrame with one row per group and aggregated non-key columns.
    """
    
    def combine_series(series):
        """Aggregate one column using its distinct, non-null values."""
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
    Uppercase and trim object-typed columns in place.

    Text values that become ``NAN``, ``NA``, ``NONE``, or ``NULL`` are replaced
    with ``pd.NA``. Other column types are left unchanged.

    Args:
        df: DataFrame whose object columns will be normalized.

    Returns:
        The same, modified DataFrame.
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
    Extract a range value from a string, mapping, list, or tuple.

    Mapping inputs use their ``"range"`` value; list/tuple values are joined
    with hyphens. Plain text is searched for a numeric range (for example,
    ``"12 to 34"``), then returned trimmed if no such range is present.

    Args:
        x: Value to inspect.

    Returns:
        A normalized range/string, or ``None`` for a missing or empty value.
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
    Lowercase text, retain alphabetic words of at least three characters, and
    remove the supplied stopwords.

    Args:
        text: Text to preprocess. Non-string values produce an empty string.
        stopwords: Words to remove, or ``None`` for no custom stopwords.

    Returns:
        The retained words joined by single spaces.
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
    """Join non-empty, non-null row values with ``"; "``; return ``pd.NA`` if none remain.

    Args:
        row: Pandas Series containing values to combine.

    Returns:
        Joined string or ``pd.NA``.
    """
    vals = (
     row.dropna()
        .astype(str)
        .map(lambda s: s.strip())
    )
    vals = [v for v in vals if v and v.lower() not in ('nan', 'none', 'na')]
    return '; '.join(vals) if vals else pd.NA


def _prepare_vectorizer_corpus(docs: list[str], vectorizer_model):
    """
    Fit a vectorizer and create token, Gensim dictionary, and bag-of-words data.

    Args:
        docs: Documents used to fit the vectorizer and build the corpus.
        vectorizer_model: Scikit-learn-style vectorizer with ``fit`` and
            ``build_tokenizer`` methods.

    Returns:
        A tuple of the vectorizer tokenizer, per-document tokens, Gensim
        dictionary, and per-document bag-of-words corpus.
    """
    vectorizer_model.fit(docs)
    tokenizer = vectorizer_model.build_tokenizer()
    tokens = [tokenizer(doc.lower()) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    return tokenizer, tokens, dictionary, corpus


def _get_topic_words(topic_model, exclude_outliers: bool = True) -> list[str]:
    """
    Flatten the words returned for topics in a BERTopic model.

    Args:
        topic_model: Fitted BERTopic-compatible model.
        exclude_outliers: Whether to omit topic ID ``-1``.

    Returns:
        Topic words in topic order, including repeated words across topics.
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
    Estimate lexical diversity across topic words using MTLD.

    Args:
        topic_model: Fitted BERTopic-compatible model.
        min_words: Return ``0.0`` unless the collected words exceed this count.

    Returns:
        MTLD for the joined topic words, or ``0.0`` when there are too few
        words or MTLD calculation fails.
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
    Compute the silhouette score for non-outlier documents.

    Args:
        embeddings: Numeric document embedding matrix.
        topics: Topic assignment for each document; ``-1`` denotes an outlier.

    Returns:
        Silhouette score, or ``-1.0`` when there are too few valid documents,
        fewer than two valid topics, or score calculation fails.
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
    Construct a BERTopic model with the project's standard UMAP and zero-shot settings.

    Args:
        vectorizer_model: Text vectorizer supplied to BERTopic.
        embedding_model: Optional embedding model passed to BERTopic.
        candidate_topics: Optional zero-shot topic labels.
        nr_topics: Optional target topic count passed to BERTopic.
        min_topic_size: Minimum size of a topic.
        random_state: Seed for the UMAP model.
        calculate_probabilities: Whether BERTopic should calculate probabilities.

    Returns:
        A configured, unfitted BERTopic model. The zero-shot minimum similarity
        is fixed at ``0.8``.
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
    Calculate the proportion of BERTopic assignments considered outliers.

    Args:
        topics: Topic assignments; ``-1``, ``-1.0``, and ``"unknown"`` count
            as outliers.

    Returns:
        Outlier count divided by assignment count, or ``0.0`` for empty input.
    """
    topics_str = [str(t) for t in topics]
    outliers = topics_str.count("-1") + topics_str.count("-1.0") + topics_str.count("unknown")
    return outliers / len(topics_str) if topics_str else 0.0


def _count_valid_topics(topics) -> int:
    """
    Count distinct topic assignments, excluding recognized outlier values.

    Args:
        topics: Topic assignments; ``-1``, ``-1.0``, and ``"unknown"`` are
            excluded.

    Returns:
        Number of distinct non-outlier topic labels.
    """
    topics_str = set(str(t) for t in topics)
    return len(topics_str - {"-1", "-1.0", "unknown"})

# functions to evaluate topic model

def calculate_coherence_score(topic_model, dictionary=None, corpus=None, tokens=None):
    """
    Score topic coherence using Gensim's normalized PMI (``c_npmi``) measure.

    Topic words are split into individual words and mapped to IDs in
    ``dictionary``. Outlier topics and words absent from the dictionary are
    excluded. A score above zero generally indicates co-occurring topic words.

    Args:
        topic_model: Fitted topic model exposing ``get_topics`` and ``get_topic``.
        dictionary: Gensim dictionary built from the source documents.
        corpus: Optional bag-of-words corpus used by Gensim.
        tokens: Tokenized source documents; required along with ``dictionary``.

    Returns:
        The scalar c_npmi coherence score, or ``0.0`` if required inputs or
        usable topics are absent or score calculation fails.

    References:
        Röder, Both, and Hinneburg (2015), *Exploring the space of topic
        coherence measures*; Bouma (2009), *Normalized (pointwise) mutual
        information in collocation extraction*.
    """
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
    """
    Measure distinctiveness of topic vocabularies.

    The score is the number of unique topic words divided by the square root
    of the total number of topic-word assignments. Higher values indicate
    more distinct words; the result is not constrained to the interval [0, 1].

    Args:
        topic_model: Fitted topic model exposing ``get_topics``.

    Returns:
        The diversity score, or ``0`` if there are no topic-word assignments.
    """
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
    """
    Iteratively adjust 2D label positions to reduce overlaps with labels/markers.

    Repulsion separates nearby labels and markers, while a weaker attraction
    draws labels toward their corresponding markers. Each iteration's movement
    is clipped to ``max_step`` per coordinate.

    Args:
        positions: Initial label coordinates with shape ``(N, 2)``.
        markers: Corresponding anchor coordinates with shape ``(N, 2)``.
        min_dist: Distance threshold for repulsion.
        steps: Number of simulation iterations.
        repulsion_strength: Multiplier for label-to-label repulsion.
        marker_repulsion_strength: Multiplier for label-to-marker repulsion.
        attraction_strength: Multiplier for attraction toward each marker.
        max_step: Maximum absolute movement per coordinate per iteration.

    Returns:
        A floating-point array of adjusted coordinates; ``positions`` is not
        modified in place.
    """

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
    Yield values from ``start`` up to, but not including, ``stop`` by ``step``.

    Args:
        start: First yielded value.
        stop: Exclusive upper bound.
        step: Increment applied after each yielded value; use a positive step
            when ``start`` is less than ``stop``.

    Yields:
        Successive values beginning at ``start``.
    """
    
    current = start
    while current < stop:
        yield current
        current += step
        
# Apply the logic only where Trace_Type is empty, with a default value of 'NA' for no matches
def find_and_replace(df, findColumn, replaceColumn, conditions, choices):
    """
    Fill empty cells in a destination column using ``numpy.select``.

    If ``replaceColumn`` does not exist it is created with empty-string values.
    Only rows where that column is exactly ``''`` are assigned. ``conditions``
    and ``choices`` are passed to ``numpy.select`` with ``'NA'`` as the default.
    ``findColumn`` is retained in the function signature but is not used.

    Args:
        df: DataFrame to modify in place.
        findColumn: Unused legacy argument.
        replaceColumn: Destination column to populate.
        conditions: Boolean conditions accepted by ``numpy.select``.
        choices: Values corresponding to ``conditions``.

    Returns:
        ``None``. The DataFrame is modified in place.
    """
    if replaceColumn not in df.columns:
        df[replaceColumn] = ''
    
    mask = df[replaceColumn] == ''
    df.loc[mask, replaceColumn] = np.select(conditions, choices, default='NA')
    
    
def fill_missing_values(df, primary_cols, alternate_cols):
    """
    Fill null primary-column values from corresponding alternate columns.

    Each alternate column is dropped after its values have been used. The
    columns are paired using ``zip``, so only pairs up to the shorter list are
    processed.
    
    Args:
        df: DataFrame to modify in place.
        primary_cols: Primary column names to fill.
        alternate_cols: Alternate column names in corresponding order.
    
    Returns:
        The modified DataFrame.
    """
    for primary, alternate in zip(primary_cols, alternate_cols):
        df[primary] = df[primary].fillna(df[alternate])
    
        df.drop(columns=alternate, inplace=True)
    return df

def run_model(docs, embeddings, vectorizer_model, notopics, clustermodel=None, embedding_model=None, candidate_topics=None):
    """
    Fit a BERTopic model and calculate its evaluation metrics.

    The combined score weights coherence by 0.70, MTLD topic diversity by
    0.15, and silhouette score by 0.15. The input vectorizer is deep-copied
    before use. ``clustermodel`` is retained in the signature but is unused.

    Args:
        docs: Documents to fit the model on.
        embeddings: Embedding matrix corresponding to ``docs``.
        vectorizer_model: Vectorizer used by BERTopic and for coherence tokens.
        notopics: Target topic count passed to BERTopic as ``nr_topics``.
        clustermodel: Unused legacy argument.
        embedding_model: Optional BERTopic embedding model.
        candidate_topics: Optional zero-shot topic labels.

    Returns:
        A two-item list: ``[results, topic_model]``. ``results`` contains one
        tuple of ``(notopics, coherence, diversity, silhouette, weighted_score)``.
    """
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
    Evaluate a zero-shot BERTopic model across repeated training folds.

    Uses 5-fold ``RepeatedKFold`` with 5 repeats (25 fitted models). The held-out
    indices are not used for scoring: the reported fold metrics are calculated
    on each fold's training documents. Vocabulary-weighted c-TF-IDF vectors
    are compared with cosine similarity to estimate structural stability.

    Args:
        docs: Documents to split and use for fitting.
        embeddings: Embedding matrix aligned row-wise with ``docs``.
        vectorizer_model: Vectorizer to copy and fit on the full document set.
        embedding_model: Optional embedding model passed to BERTopic.
        topic_list: Candidate zero-shot topic labels.

    Returns:
        Dictionary with mean/std outlier ratio, mean topic count, mean
        structural stability, and per-fold outlier ratios/topic counts.
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
    
def rename_key(item, key_mapping):
    """
    Rename keys in a string representation of a dictionary or list of dictionaries.
    
    The function first quotes unquoted string values where possible, applies
    each old-to-new key mapping, then parses the result with
    ``ast.literal_eval``. Non-string, empty, or ``"NA"`` values pass through.

    Args:
        item: Value containing serialized mapping data.
        key_mapping: Mapping from old key names to replacement key names.

    Returns:
        The parsed Python value if parsing succeeds; otherwise the cleaned
        string. Values that are not processed are returned unchanged.
    """
    if not isinstance(item, str) or item.strip() == "NA" or not item.strip():
        return item
    
    clean_item = item.strip()
    clean_item = re.sub(
        r':\s*(?!(?:true|false|null|True|False|None)\b)([a-zA-Z_][a-zA-Z0-9_\-]*)\b(?!\s*[:\]}])', 
        r': "\1"', 
        clean_item
        )
    
    for old_key, new_key in key_mapping.items():
        pattern = rf'([\'"]?){re.escape(old_key)}\1\s*:\s*'
        clean_item = re.sub(pattern, rf'\1{new_key}\1: ', clean_item)
    

    try:
        parsed_list = ast.literal_eval(clean_item)
        return parsed_list
    except (SyntaxError, ValueError):
        return clean_item


def lib_to_dict(entry: Entry) -> dict:
    """Convert a bibtexparser entry into a dictionary of its key, type, and fields.

    Args:
        entry: bibtexparser entry exposing ``key``, ``entry_type``, and
            ``fields_dict`` attributes.

    Returns:
        A dictionary containing ``ID``, ``ENTRYTYPE``, and all entry fields.
    """
    entry_data = {
        "ID": entry.key,
        "ENTRYTYPE": entry.entry_type,
        **entry.fields_dict
    }
    return entry_data


def convert_fields(data: pd.Series) -> list:
    """Extract field values from a Series of bibtexparser fields.

    Args:
        data: Series whose elements may be bibtexparser ``Field`` instances.

    Returns:
        A list containing each field's ``value`` or ``pd.NA`` for non-Field
        elements.
    """
    newContent = [field.value if isinstance(field, bibtexparser.model.Field) else pd.NA for field in data]
    return newContent

def remove_duplicate_dicts(input_data: list[dict]) -> list[dict]:
    """Flatten nested lists of dictionaries and deduplicate them in input order.

    Dictionary values containing nested dictionaries, lists, or tuples are
    recursively converted to hashable forms for equality checks. Non-dictionary
    items are ignored.

    Args:
        input_data: Iterable list structure containing dictionaries and nested
            lists.

    Returns:
        A flat list of unique dictionaries, preserving the first occurrence.
    """
    # Step 1: Flatten all nested lists of dicts
    flat_list = []
    
    def flatten(items):
        """Append dictionaries from nested lists to ``flat_list``."""
        for item in items:
            if isinstance(item, list):
                flatten(item)  # Recursively unwrap lists
            elif isinstance(item, dict):
                flat_list.append(item)
            # If there are other types (like strings), they are skipped
            
    flatten(input_data)

    # Step 2: Helper to deep-freeze mutable types so they can be hashed
    def make_hashable(obj):
        """Recursively convert dictionary/list/tuple values to hashable forms."""
        if isinstance(obj, dict):
            return frozenset((k, make_hashable(v)) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return tuple(make_hashable(item) for item in obj)
        return obj

    # Step 3: Deduplicate while preserving order
    seen = set()
    unique_list = []
    
    for d in flat_list:
        track_key = make_hashable(d)
        if track_key not in seen:
            seen.add(track_key)
            unique_list.append(d)
            
    return unique_list

def separate_author_initials(authors):
    """Insert separators between detected author initials and following names.

    Initial-like tokens are separated with a comma, except when the following
    token indicates a surname particle according to the module's
    ``surname_particles`` collection.

    Args:
        authors: Author string to process.

    Returns:
        The author string with separators inserted at detected initials.
    """
    def add_separator(match):
        """Build a replacement for one initials-like regular-expression match."""
        following_token = re.match(r"\w+", authors[match.end():])
        if match.group(2).lower() in surname_particles and following_token:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}, "

    return re.sub(r"(\s+)((?:[A-Z]{1,2}|[A-Z][a-z]{1,2}))\b\s*", add_separator, authors)

def build_nested_groups(dataframe, keys_remaining, combine_func, rows_list=None):
    """Recursively group a DataFrame across keys while preserving missing-key rows.

    At each key, non-null values are grouped separately and rows with a null
    key are processed together as their own branch. At the leaf level,
    ``combine_func`` is applied once to each non-empty subset.

    Args:
        dataframe: DataFrame to split into nested groups.
        keys_remaining: Ordered column names to group by at each recursion level.
        combine_func: Callable applied to each non-empty leaf DataFrame.
        rows_list: Optional list to append results to; a new list is created
            when omitted.

    Returns:
        The accumulated list of results returned by ``combine_func``.
    """
    if rows_list is None:
        rows_list = []
        
    # BASE CASE: When we reach the final level, apply the lambda and save the row
    if not keys_remaining:
        if not dataframe.empty:
            single_row_series = combine_func(dataframe)
            rows_list.append(single_row_series)
        return rows_list
    
    current_key = keys_remaining[0]
    next_keys = keys_remaining[1:]
    
    valid_mask = dataframe[current_key].notna()
    valid_df = dataframe[valid_mask]
    missing_df = dataframe[~valid_mask]
    
    # Process valid groups
    for group_name, group_df in valid_df.groupby(current_key):
        build_nested_groups(group_df, next_keys, combine_func, rows_list)
        
    # Process missing group explicitly
    if not missing_df.empty:
        build_nested_groups(missing_df, next_keys, combine_func, rows_list)
        
    return rows_list
