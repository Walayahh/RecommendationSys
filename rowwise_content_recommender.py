#############################################
# rowwise content recommender
#############################################
"""
Implements a row-by-row top-k approach for content-based recommendations
using TF-IDF, to avoid constructing a huge NxN similarity matrix.

Flow:
  1) Read perfumes.csv, combine textual features.
  2) Build TF-IDF: shape (N, M).
  3) For each user-selected perfume:
     a) Multiply that perfume's TF-IDF row vector by tfidf_matrix.T (N x 1).
     b) Get top_k neighbors from that single row (excl. itself).
     c) Combine (sum) scores if multiple items are selected.
  4) Return final top-n recommendations.
No MemoryError from NxN because we never do a full 'cosine_similarity(tfidf, tfidf)'.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

#############################################
# 1) READ & PREPARE THE DATA
#############################################
def load_and_prepare_perfumes(csv_path, delimiter=";", encoding="latin-1"):
    logger.info("Loading perfumes data from %s", csv_path)
    df = pd.read_csv(csv_path, delimiter=delimiter, encoding=encoding)

    # Ensure we have columns: 'Top', 'Middle', 'Base', 'mainaccord1'...'mainaccord5', etc.
    # If you have a different structure, adjust accordingly.
    df["combined_features"] = (
        df["Top"].fillna("") + " " +
        df["Middle"].fillna("") + " " +
        df["Base"].fillna("") + " " +
        df["mainaccord1"].fillna("") + " " +
        df["mainaccord2"].fillna("") + " " +
        df["mainaccord3"].fillna("") + " " +
        df["mainaccord4"].fillna("") + " " +
        df["mainaccord5"].fillna("")
    )

    logger.info("Perfumes loaded with shape: %s", df.shape)
    return df

#############################################
# 2) TF-IDF and Indices
#############################################
def build_tfidf_matrix(df, text_col="combined_features"):
    """
    Build a TF-IDF matrix from the DataFrame's 'text_col' column.
    Returns:
      - tfidf_matrix (scipy sparse NxM)
      - index_to_name (list of perfume names)
      - name_to_index (dict for lookup)
    """
    documents = df[text_col].fillna("").values
    logger.info("Building TF-IDF for %d documents...", len(documents))

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)

    # Build index <-> perfume name mapping
    index_to_name = df["Perfumes"].tolist()
    name_to_index = {name: i for i, name in enumerate(index_to_name)}
    #print("name to index",name_to_index)

    return tfidf_matrix, index_to_name, name_to_index

#############################################
# 3) ROW-BY-ROW TOP-K FUNCTION
#############################################
def get_topk_for_perfume(
    perfume_name,
    tfidf_matrix,
    name_to_index,
    index_to_name,
    top_k=50
):
    """
    Given a single 'perfume_name', compute similarity row:
      row = tfidf_matrix[row_i] dot tfidf_matrix.T => shape (N,)
    Then pick top_k results (excluding itself).
    Returns a list of (perfume_name, score).
    """
    perfume_name_lower = perfume_name.lower()
    # 1) Find the index
    # We do a direct match. If there's a mismatch in naming, we'll skip
    matched_index = None
    for real_name, i in name_to_index.items():
        if real_name.lower() == perfume_name_lower:
            matched_index = i
            break
    if matched_index is None:
        logger.warning("Perfume '%s' not found in name_to_index. Returning empty.", perfume_name)
        return []

    # 2) Dot product row
    # row_i => shape (1, M)
    row_i = tfidf_matrix[matched_index]
    # sim_vec => shape (1, N) after row_i * tfidf_matrix^T => we get dense or sparse?
    # We'll do .toarray() so we can manipulate it easily, but we must watch memory usage.
    sim_vec = row_i.dot(tfidf_matrix.T).toarray().ravel()  # shape (N,)

    # 3) Exclude itself
    sim_vec[matched_index] = -999.0

    # 4) find top_k via partial sort
    # We'll gather the top_k indices, then create a list (perf_name, score).
    if top_k >= len(sim_vec):
        # if top_k is bigger than the dataset, just sort the entire array
        top_indices = np.argsort(sim_vec)[::-1][:top_k]
    else:
        # partial selection
        # or we can do 'np.argpartition(sim_vec, -top_k)' if we want partial
        # for simplicity, let's do that
        partition_idx = np.argpartition(sim_vec, -top_k)[-top_k:]
        top_indices = partition_idx[np.argsort(sim_vec[partition_idx])[::-1]]

    results = [(index_to_name[idx], sim_vec[idx]) for idx in top_indices]
    return results

#############################################
# 4) AGGREGATE MULTIPLE SELECTED PERFUMES
#############################################
def get_content_based_recommendations(
    selected_perfumes,
    tfidf_matrix,
    name_to_index,
    index_to_name,
    top_k_each=50,
    final_n=5
):
    """
    If the user selected multiple perfumes, we:
      - For each selected perfume, get top_k_each neighbors.
      - Sum up similarity scores across those sets.
      - Sort & pick the final_n best suggestions (excluding the user selected).
    Returns a list of recommended perfume names.
    """
    logger.info("Starting content-based rowwise aggregator for %s", selected_perfumes)
    agg_scores = {}
    selected_lower = {p.lower() for p in selected_perfumes}

    for perfume in selected_perfumes:
        logger.info("Fetching top-%d for '%s'...", top_k_each, perfume)
        topk_list = get_topk_for_perfume(
            perfume,
            tfidf_matrix,
            name_to_index,
            index_to_name,
            top_k=top_k_each
        )
        # topk_list => [(name, score), (name2, score2), ...]
        for (name, score) in topk_list:
            if name.lower() in selected_lower:
                # skip if user already selected it
                continue
            agg_scores[name] = agg_scores.get(name, 0.0) + score

    if not agg_scores:
        logger.warning("No scores aggregated. Possibly no matches or mismatch in naming.")
        return []

    # Sort by descending aggregated score
    sorted_items = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)
    final_recs = [name for (name, score) in sorted_items[:final_n]]
    logger.info("Final recommendations => %s", final_recs)
    return final_recs