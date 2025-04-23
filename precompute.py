###################################
# precompute.py
###################################
"""
This script:
  1) Computes and caches the content-based similarity matrix (content_sim.pkl).
  2) Computes and caches the item-based collaborative similarity matrix (collab_sim.pkl).
  3) (Optional) Combines both to produce a hybrid matrix (combined_sim.pkl) if desired.

Run:
  python precompute.py
to generate or update the cached similarity files. Set force_recompute=True in the 
load_or_compute_*() calls if you want to ignore existing caches and recompute everything.

Usage example:
  1) Make sure 'perfumes.csv' (semicolon-delimited) and 'users.csv' (comma-delimited)
     are in the expected paths (or adjust paths below).
  2) python precompute.py
  3) The content-based matrix is saved to 'content_sim.pkl'.
     The collaborative matrix is saved to 'collab_sim.pkl'.
     The combined matrix is saved to 'combined_sim.pkl' if do_hybrid=True.
"""

import logging
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# FILENAMES FOR CACHING
# -------------------------------------------------------------------------
CONTENT_SIM_FILE = "content_sim.pkl"
COLLAB_SIM_FILE = "collab_sim.pkl"
COMBINED_SIM_FILE = "combined_sim.pkl"

# -------------------------------------------------------------------------
# 1) CONTENT-BASED SIMILARITY
# -------------------------------------------------------------------------
def prepare_content_based_similarity(df_perfumes):
    """
    Build a content-based similarity matrix from perfume data:
      - Combine text fields (Top, Middle, Base notes, accords, perfumer).
      - TF-IDF vectorize, then compute cosine similarity.
    Returns a DataFrame content_sim with shape [N x N], index/columns = perfume names.
    """
    logger.info("Starting prepare_content_based_similarity...")

    df_perfumes['combined_features'] = (
        df_perfumes['Top'].fillna('') + ' ' +
        df_perfumes['Middle'].fillna('') + ' ' +
        df_perfumes['Base'].fillna('') + ' ' +
        df_perfumes['mainaccord1'].fillna('') + ' ' +
        df_perfumes['mainaccord2'].fillna('') + ' ' +
        df_perfumes['mainaccord3'].fillna('') + ' ' +
        df_perfumes['mainaccord4'].fillna('') + ' ' +
        df_perfumes['mainaccord5'].fillna('') + ' ' +
        df_perfumes['Perfumer1'].fillna('') + ' ' +
        df_perfumes['Perfumer2'].fillna('')
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_perfumes['combined_features'])
    logger.info("TF-IDF matrix shape (content): %s", tfidf_matrix.shape)

    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logger.info("Content-based similarity matrix shape: %s", sim_matrix.shape)

    content_sim = pd.DataFrame(
        sim_matrix,
        index=df_perfumes['Perfume'],
        columns=df_perfumes['Perfume']
    )
    logger.info("Finished prepare_content_based_similarity.")
    return content_sim

def load_or_compute_content_sim(df_perfumes, force_recompute=False):
    """
    Loads content_sim.pkl if it exists (and force_recompute=False).
    Otherwise, computes it from df_perfumes, then saves to disk.
    """
    if not force_recompute and os.path.exists(CONTENT_SIM_FILE):
        logger.info("Loading content-based sim from %s", CONTENT_SIM_FILE)
        with open(CONTENT_SIM_FILE, "rb") as f:
            content_sim = pickle.load(f)
    else:
        logger.info("No valid content_sim cache or force_recompute=True. Computing fresh content-based matrix...")
        content_sim = prepare_content_based_similarity(df_perfumes)
        logger.info("Saving new content-based matrix to %s", CONTENT_SIM_FILE)
        with open(CONTENT_SIM_FILE, "wb") as f:
            pickle.dump(content_sim, f)
    return content_sim

# -------------------------------------------------------------------------
# 2) ITEM-BASED COLLABORATIVE SIMILARITY
# -------------------------------------------------------------------------
def prepare_item_based_similarity(df_users, df_perfumes):
    """
    Build an item-based CF matrix from user data:
      - Create user x item matrix (0/1).
      - Transpose to item x user.
      - Cosine similarity among items (perfumes).
    Returns collab_sim (DataFrame).
    """
    logger.info("Starting prepare_item_based_similarity...")

    all_perfume_names = df_perfumes['Perfume'].unique().tolist()
    logger.info("Found %d unique perfumes in df_perfumes.", len(all_perfume_names))

    # user->set_of_perfumes
    user_perfume_map = {}
    for idx, row in df_users.iterrows():
        raw_str = str(row.get('Perfumes', ''))
        raw_list = raw_str.split(',')
        chosen = {p.strip().lower() for p in raw_list if p.strip()}
        user_perfume_map[idx] = chosen

    # Create user x item matrix
    user_ids = list(user_perfume_map.keys())
    user_item_matrix = pd.DataFrame(data=0, index=user_ids, columns=all_perfume_names)
    name_map = {p.lower(): p for p in all_perfume_names}

    for user_id, chosen_set in user_perfume_map.items():
        for cp in chosen_set:
            if cp in name_map:
                real_name = name_map[cp]
                user_item_matrix.at[user_id, real_name] = 1

    # Transpose => item x user
    item_user_matrix = user_item_matrix.T
    logger.info("item_user_matrix shape: %s", item_user_matrix.shape)

    sim_matrix = cosine_similarity(item_user_matrix, item_user_matrix)
    logger.info("Collaborative similarity matrix shape: %s", sim_matrix.shape)

    collab_sim = pd.DataFrame(
        sim_matrix,
        index=all_perfume_names,
        columns=all_perfume_names
    )
    logger.info("Finished prepare_item_based_similarity.")
    return collab_sim

def load_or_compute_collab_sim(df_users, df_perfumes, force_recompute=False):
    """
    Loads collab_sim.pkl if it exists (and force_recompute=False).
    Otherwise, computes item-based CF from df_users, df_perfumes, then saves to disk.
    """
    if not force_recompute and os.path.exists(COLLAB_SIM_FILE):
        logger.info("Loading collaborative sim from %s", COLLAB_SIM_FILE)
        with open(COLLAB_SIM_FILE, "rb") as f:
            collab_sim = pickle.load(f)
    else:
        logger.info("No valid collab_sim cache or force_recompute=True. Computing fresh collaborative matrix...")
        collab_sim = prepare_item_based_similarity(df_users, df_perfumes)
        logger.info("Saving new collaborative matrix to %s", COLLAB_SIM_FILE)
        with open(COLLAB_SIM_FILE, "wb") as f:
            pickle.dump(collab_sim, f)
    return collab_sim

# -------------------------------------------------------------------------
# 3) COMBINED (HYBRID) SIMILARITY
# -------------------------------------------------------------------------
def prepare_combined_similarity(content_sim, collab_sim, alpha=0.5):
    """
    Weighted sum: alpha * collab_sim + (1 - alpha) * content_sim.
    Perfumes must be present in both index/columns => intersection used.
    """
    logger.info("Starting prepare_combined_similarity with alpha=%.2f...", alpha)

    common_perfumes = content_sim.index.intersection(collab_sim.index)
    content_sim = content_sim.loc[common_perfumes, common_perfumes]
    collab_sim = collab_sim.loc[common_perfumes, common_perfumes]

    combined = alpha * collab_sim + (1 - alpha) * content_sim
    logger.info("Combined sim shape: %s", combined.shape)
    logger.info("Finished prepare_combined_similarity.")
    return combined

def load_or_compute_combined_sim(alpha=0.5, force_recompute=False):
    """
    Loads combined_sim.pkl if it exists and force_recompute=False.
    Otherwise, loads content_sim and collab_sim from pkl (or compute them),
    performs the hybrid combination, and saves combined_sim.pkl.

    This function assumes content_sim.pkl, collab_sim.pkl were already computed
    or can be automatically computed if missing (just ensure you run the 
    load_or_compute_* functions for them).
    """
    if not force_recompute and os.path.exists(COMBINED_SIM_FILE):
        logger.info("Loading combined sim from %s", COMBINED_SIM_FILE)
        with open(COMBINED_SIM_FILE, "rb") as f:
            combined_sim = pickle.load(f)
        return combined_sim

    logger.info("No valid combined_sim cache or force_recompute=True. Will combine content & collab now...")

    # We assume content_sim.pkl and collab_sim.pkl exist or can be loaded
    with open(CONTENT_SIM_FILE, "rb") as f:
        content_sim = pickle.load(f)
    with open(COLLAB_SIM_FILE, "rb") as f:
        collab_sim = pickle.load(f)

    combined_sim = prepare_combined_similarity(content_sim, collab_sim, alpha)
    logger.info("Saving new combined sim to %s", COMBINED_SIM_FILE)
    with open(COMBINED_SIM_FILE, "wb") as f:
        pickle.dump(combined_sim, f)
    return combined_sim

# -------------------------------------------------------------------------
# 4) RECOMMENDATION FUNCTION (for demonstration)
# -------------------------------------------------------------------------
def get_item_based_recommendations(selected_perfumes, combined_sim, top_n=5):
    """
    Summation-based approach: for each user-selected perfume, add up similarity 
    scores across the combined_sim matrix. Sort, pick top_n.
    """
    logger.info("Starting get_item_based_recommendations for %s, top_n=%d", selected_perfumes, top_n)

    agg_scores = {}
    all_perfumes = combined_sim.index.tolist()
    lower_to_real = {p.lower(): p for p in all_perfumes}

    selected_lower = [p.lower() for p in selected_perfumes if isinstance(p, str)]

    for p in selected_lower:
        if p in lower_to_real:
            real_p = lower_to_real[p]
            row_sim = combined_sim.loc[real_p]
            for perfume_name, sim_val in row_sim.items():
                if perfume_name.lower() in selected_lower:
                    continue
                agg_scores[perfume_name] = agg_scores.get(perfume_name, 0) + sim_val

    # Sort by descending total similarity
    sorted_perfumes = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)
    top_recs = [perf for perf, score in sorted_perfumes[:top_n]]

    logger.info("Final recommendations: %s", top_recs)
    logger.info("Finished get_item_based_recommendations.")
    return top_recs

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Running precompute.py to build/cache similarity matrices...")

    # 1) Load your CSV data
    df_perfumes = pd.read_csv("perfumes.csv", delimiter=";", encoding="latin-1")


# Define a function to replace dashes with spaces and convert to title case
    def transform(text):
        if isinstance(text, str):
            return text.replace("-", " ").title()
        return text

    # Apply the transformation to both the 'Perfume' and 'Brand' columns
    df_perfumes["Perfume"] = df_perfumes["Perfume"].apply(transform)
    df_perfumes["Brand"] = df_perfumes["Brand"].apply(transform)
    df_perfumes["Perfumes"] = df_perfumes["Brand"].apply(transform) + " " +  df_perfumes["Perfume"].apply(transform)
    df_users = pd.read_csv("users.csv", delimiter=",")
    logger.info("Loaded df_perfumes shape: %s, df_users shape: %s", df_perfumes.shape, df_users.shape)

    # 2) Compute or load content-based sim => content_sim.pkl
    content_sim = load_or_compute_content_sim(df_perfumes, force_recompute=False)

    # 3) Compute or load collab-based sim => collab_sim.pkl
    collab_sim = load_or_compute_collab_sim(df_users, df_perfumes, force_recompute=False)

    # 4) Optionally combine them => combined_sim.pkl
    #    If you only need them separate, you can skip this.
    do_hybrid = True
    alpha = 0.65
    if do_hybrid:
        combined_sim = load_or_compute_combined_sim(alpha=alpha, force_recompute=False)

        # Show a quick example of retrieving recs from the combined matrix
        user_selected = ["le-male-pride-2023", "classique-pride-2024"]
        logger.info("User selected (demo): %s", user_selected)
        # Actually get recommendations from combined
        recs = get_item_based_recommendations(user_selected, combined_sim, top_n=5)
        logger.info("Recommendations => %s", recs)

    logger.info("Done. Matrices are saved. You can now use them in your Flask app or other scripts.")
