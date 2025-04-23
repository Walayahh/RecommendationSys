#############################################
# matrix_factorization_recommender.py
# -------------------------------------------
# Implicit‑feedback Matrix Factorisation via
# custom ALS in NumPy (no external recommender libs).
# Uses your binary "like" CSV data to factorise
# into latent factors with user/item biases.
#############################################

import os
import pickle
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# Configure logger
glogger = logging.getLogger(__name__)
glogger.setLevel(logging.INFO)

# Path to cache the trained model
MODEL_PATH = "als_numpy.pkl"

###############################################################################
# 1. BUILD INTERACTION DICTS
###############################################################################

def build_interaction_dicts(
    users_csv: str,
    perfumes_csv: str,
    user_col: str = "UserID",
    delimiter_users: str = ",",
    delimiter_perfumes: str = ";",
    encoding_users: str = "latin-1",
    encoding_perfumes: str = "latin-1",
) -> Tuple[
    Dict[int, List[int]],  # user2perfumes
    Dict[int, List[int]],  # perfume2users
    Dict[Tuple[int,int], float],  # userperfume2rating
    Dict[str,int],         # perfume2id
    Dict[int,str]          # id2perfume
]:
    """
    Parse CSVs to dicts:
      - user2perfumes: uid -> [pid,...]
      - perfume2users: pid -> [uid,...]
      - userperfume2rating: (uid,pid)->1.0
      - perfume2id, id2perfume mappings
    """
    # load users
    df_users = pd.read_csv(users_csv, delimiter=delimiter_users, encoding=encoding_users)
    if user_col not in df_users.columns:
        df_users[user_col] = df_users.index
        glogger.warning("'%s' missing: using row index as user ID", user_col)

    # load perfumes
    df_perf = pd.read_csv(perfumes_csv, delimiter=delimiter_perfumes, encoding=encoding_perfumes)
    unique_perfs = df_perf['Perfumes'].unique().tolist()
    perfume2id = {name: i for i, name in enumerate(unique_perfs)}
    id2perfume = {i: name for name, i in perfume2id.items()}
    
    user2perfumes = {}
    perfume2users = {}
    userperfume2rating = {}

    for _, row in df_users.iterrows():
        uid = int(row[user_col])
        chosen = [p.strip() for p in str(row.get('Perfumes','')).split(',') if p.strip()]
        for pname in chosen:
            pid = perfume2id.get(pname)
            if pid is None:
                continue
            user2perfumes.setdefault(uid, []).append(pid)
            perfume2users.setdefault(pid, []).append(uid)
            userperfume2rating[(uid, pid)] = 1.0


    if not userperfume2rating:
        raise ValueError("No interactions found in users CSV")

    return user2perfumes, perfume2users, userperfume2rating, perfume2id, id2perfume

###############################################################################
# 2. NUMPY ALS TRAINING
###############################################################################

def als_train(
    user2perfumes: Dict[int, List[int]],
    perfume2users: Dict[int, List[int]],
    userperfume2rating: Dict[Tuple[int,int], float],
    n_users: int,
    n_items: int,
    K: int = 50,
    reg: float = 0.1,
    n_iter: int = 20,
    verbose: bool = True
) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    """
    Factorise the implicit feedback binary matrix via
    Alternating Least Squares with biases.
    Returns U, V, b, c, mu.
    """
    # init
    U  = 0.01 * np.random.randn(n_users, K)
    V  = 0.01 * np.random.randn(n_items, K)
    b  = np.zeros(n_users)
    c  = np.zeros(n_items)
    mu = np.mean(list(userperfume2rating.values()))
    I  = np.eye(K)

    for it in range(1, n_iter+1):
        # users
        for u, pids in user2perfumes.items():
            V_I = V[pids]
            r_I = np.array([userperfume2rating[(u,p)] for p in pids])
            c_I = c[pids]
            A = V_I.T@V_I + reg*I
            rhs = ((r_I - b[u] - c_I - mu)[:,None]*V_I).sum(axis=0)
            U[u] = np.linalg.solve(A, rhs)
        # items
        for p, uids in perfume2users.items():
            U_J = U[uids]
            r_J = np.array([userperfume2rating[(u,p)] for u in uids])
            b_J = b[uids]
            A = U_J.T@U_J + reg*I
            rhs = ((r_J - b_J - c[p] - mu)[:,None]*U_J).sum(axis=0)
            V[p] = np.linalg.solve(A, rhs)
        # biases
        for u, pids in user2perfumes.items():
            preds = U[u]@V[pids].T
            resid = np.array([userperfume2rating[(u,p)] for p in pids]) - preds - c[pids] - mu
            b[u] = resid.sum()/(len(pids)+reg)
        for p, uids in perfume2users.items():
            preds = U[uids]@V[p].T
            resid = np.array([userperfume2rating[(u,p)] for u in uids]) - preds - b[uids] - mu
            c[p] = resid.sum()/(len(uids)+reg)
        if verbose:
            print(f"ALS iter {it}/{n_iter} done")
    return U, V, b, c, mu

###############################################################################
# 3. LOAD / TRAIN WITH PICKLE
###############################################################################

def load_or_train_als(
    users_csv: str,
    perfumes_csv: str,
    factors: int = 50,
    reg: float = 0.1,
    n_iter: int = 20,
    force_retrain: bool = False
) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float,Dict[str,int],Dict[int,str]]:
    """
    Load pickled ALS factors or train anew.
    Returns (U,V,b,c,mu, perfume2id, id2perfume).
    """
    if os.path.exists(MODEL_PATH) and not force_retrain:
        glogger.info("Loading ALS factors from %s", MODEL_PATH)
        return pickle.load(open(MODEL_PATH, 'rb'))
    # build dicts
    user2perfumes, perfume2users, up2r, p2id, id2p = build_interaction_dicts(
        users_csv, perfumes_csv
    )
    n_users = max(user2perfumes.keys())+1
    n_items = max(perfume2users.keys())+1
    # train
    print(f"Training ALS: factors={factors}, reg={reg}, iters={n_iter}")
    U, V, b, c, mu = als_train(
        user2perfumes, perfume2users, up2r,
        n_users, n_items, factors, reg, n_iter
    )
    # pickle
    pickle.dump((U, V, b, c, mu, p2id, id2p), open(MODEL_PATH, 'wb'))
    print("ALS factors saved to %s", MODEL_PATH)
    return U, V, b, c, mu, p2id, id2p

###############################################################################
# 4. RECOMMEND SIMILAR ITEMS
###############################################################################

def recommend_similar(
    V: np.ndarray,
    id2perfume: Dict[int,str],
    perfume2id: Dict[str,int],
    target: str,
    K: int = 10
) -> List[str]:
    """
    Simple item-item via latent dot product.
    """
    pid = perfume2id.get(target)
    if pid is None:
        glogger.warning("'%s' not recognized", target)
        return []
    vec = V[pid]
    scores = V @ vec
    scores[pid] = -np.inf
    top = np.argpartition(scores, -K)[-K:]
    top_sorted = top[np.argsort(scores[top])[::-1]]
    return [id2perfume[i] for i in top_sorted]

###############################################################################
# 5. MAIN: TRAIN & DUMP
###############################################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train ALS CSV pipeline")
    parser.add_argument('--users', default='users.csv')
    parser.add_argument('--perfumes', default='perfumes_updated.csv')
    parser.add_argument('--factors', type=int, default=50)
    parser.add_argument('--reg', type=float, default=0.1)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    load_or_train_als(
        args.users,
        args.perfumes,
        factors=args.factors,
        reg=args.reg,
        n_iter=args.iters,
        force_retrain=args.force
    )
    glogger.info("Done.")