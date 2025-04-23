#############################################
# rowwise_collab_recommender.py
#############################################

import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_item_user_matrix(
    users_csv,
    perfumes_csv,
    user_col="UserID",
    delimiter_users=",",
    delimiter_perfumes=";",
    encoding_users="latin-1",      
    encoding_perfumes="latin-1"   
):
    """
    Reads 'users.csv' and 'perfumes_updated.csv' with specified encodings.
    Builds item_user matrix in row-by-row approach.
    """
    logger.info("Loading user data from %s", users_csv)
    df_users = pd.read_csv(users_csv, delimiter=delimiter_users, encoding=encoding_users)

    if user_col not in df_users.columns:
        logger.warning("No '%s' column found in users CSV. We'll use row index as user ID.", user_col)
        df_users[user_col] = df_users.index

    user_id_to_perfumes = {}
    for idx, row in df_users.iterrows():
        user_id = row[user_col]
        raw_list = str(row.get("Perfumes", "")).split(",")
        chosen = {p.strip().lower() for p in raw_list if p.strip()}
        user_id_to_perfumes[user_id] = chosen

    logger.info("Loading perfume data from %s", perfumes_csv)
    df_perfumes = pd.read_csv(perfumes_csv, delimiter=delimiter_perfumes, encoding=encoding_perfumes)

    all_perfume_names = df_perfumes["Perfumes"].unique().tolist()
    perfume_lower_to_original = {p.lower(): p for p in all_perfume_names}

    logger.info("Building user-item matrix. #users=%d, #perfumes=%d", len(user_id_to_perfumes), len(all_perfume_names))

    item_names = sorted(all_perfume_names)
    user_names = sorted(user_id_to_perfumes.keys())

    # Create user-item (#users x #perfumes) filled with 0
    user_item_df = pd.DataFrame(
        data=0,
        index=user_names,
        columns=item_names,
        dtype=np.float32
    )

    # Fill 1 if user selected that perfume
    for user_id, chosen_set in user_id_to_perfumes.items():
        for c in chosen_set:
            if c in perfume_lower_to_original:
                real_name = perfume_lower_to_original[c]
                if real_name in user_item_df.columns:
                    user_item_df.at[user_id, real_name] = 1.0

    # Transpose => #perfumes x #users
    item_user_df = user_item_df.T
    logger.info("item_user matrix shape = %s", item_user_df.shape)

    item_to_index = {name: i for i, name in enumerate(item_user_df.index)}
    return item_user_df, item_user_df.index.tolist(), item_user_df.columns.tolist(), item_to_index


def get_topk_for_item(item_name, item_user_df, item_to_index, top_k=50):
    item_name_lower = item_name.lower()
    matched_name = None
    for real_name, idx in item_to_index.items():
        if real_name.lower() == item_name_lower:
            matched_name = real_name
            break
    if matched_name is None:
        logging.warning("Item '%s' not found. Returning [].", item_name)
        return []

    row_i = item_user_df.loc[matched_name].values.reshape(1, -1)
    matrix_values = item_user_df.values
    sim_vec = row_i @ matrix_values.T
    sim_vec = sim_vec.ravel()

    self_idx = item_to_index[matched_name]
    sim_vec[self_idx] = -999.0

    n_items = len(sim_vec)
    if top_k >= n_items:
        top_indices = np.argsort(sim_vec)[::-1][:top_k]
    else:
        partition_idx = np.argpartition(sim_vec, -top_k)[-top_k:]
        top_indices = partition_idx[np.argsort(sim_vec[partition_idx])[::-1]]

    all_names = item_user_df.index
    results = [(all_names[idx], float(sim_vec[idx])) for idx in top_indices]
    return results


def get_collab_recommendations(selected_items, item_user_df, item_to_index, top_k_each=50, final_n=5):
    logger.info("Starting item-based collab aggregator for %s", selected_items)
    agg_scores = {}
    selected_lower = {si.lower() for si in selected_items}

    for itm in selected_items:
        topk_neighbors = get_topk_for_item(itm, item_user_df, item_to_index, top_k=top_k_each)
        for (neighbor_name, score) in topk_neighbors:
            if neighbor_name.lower() in selected_lower:
                continue
            agg_scores[neighbor_name] = agg_scores.get(neighbor_name, 0.0) + score

    if not agg_scores:
        logger.warning("No scores :( ")
        return []

    sorted_items = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)
    recs = [name for (name, score) in sorted_items[:final_n]]
    logger.info("Final recommendations => %s", recs)
    return recs
