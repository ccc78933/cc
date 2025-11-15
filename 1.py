# app.py  —— FastAPI + 协同过滤新闻推荐（ItemCF + ALS）
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from scipy.sparse import coo_matrix
import implicit

# =========================
# 配置
# =========================
TAU_DAYS = 5
ALS_FACTORS = 64
ALS_ALPHA = 40
ALS_REG = 1e-2
ALS_ITERS = 15
FRESH_HALF_LIFE_H = 24
MAX_AGE_DAYS = 7
ICF_ALPHA = 0.5
MERGE_W_ICF = 0.5
MERGE_W_ALS = 0.7
MMR_LAMBDA = 0.85
TOPK_RETURN = 50
ICF_RECALL = 300
ALS_RECALL = 300

# =========================
# 全局状态（内存）
# =========================
item_meta: Dict[str, Dict] = {}
interactions_df: pd.DataFrame = None
by_user: Dict[str, List[Tuple[str, float]]] = {}
item_sim: Dict[str, Dict[str, float]] = {}
user2idx: Dict[str, int] = {}
item2idx: Dict[str, int] = {}
idx2item: Dict[int, str] = {}
S = None
als_model = None
now_ts = datetime.now(timezone.utc).timestamp()

# =========================
# 工具
# =========================
def implicit_weight(evt: dict, now: float = None, tau_days: int = TAU_DAYS) -> float:
    if now is None:
        now = datetime.now(timezone.utc).timestamp()
    base = (
        float(evt.get("click", 0)) * 1.0
        + float(evt.get("finish", 0)) * 0.5
        + float(evt.get("like", 0)) * 1.0
        + float(evt.get("collect", 0)) * 1.0
        + float(evt.get("comment", 0)) * 1.0
    )
    decay = math.exp(-(now - float(evt["ts"])) / (tau_days * 86400))
    return base * decay

def freshness_boost(score: float, publish_ts: float, now_ts: float, half_life_h: int = FRESH_HALF_LIFE_H) -> float:
    decay = math.exp(-(now_ts - publish_ts) / (half_life_h * 3600))
    return 0.7 * score + 0.3 * decay

def too_old(publish_ts: float, now_ts: float, days: int = MAX_AGE_DAYS) -> bool:
    return (now_ts - publish_ts) > days * 86400

# =========================
# 数据加载
# =========================
def load_items(path: str) -> Dict[str, Dict]:
    df = pd.read_csv(path)
    meta = {}
    for _, r in df.iterrows():
        meta[str(r["item_id"])] = {
            "publish_ts": float(r["publish_ts"]),
            "category": str(r.get("category", "")),
            "source": str(r.get("source", "")),
        }
    return meta

def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["click", "finish", "like", "collect", "comment"]:
        if col not in df.columns:
            df[col] = 0
    return df

# =========================
# ItemCF
# =========================
def build_user_item_weights(df: pd.DataFrame, now: float) -> Dict[str, List[Tuple[str, float]]]:
    by_u = defaultdict(list)
    for _, r in df.iterrows():
        evt = {
            "click": r.get("click", 0),
            "finish": r.get("finish", 0),
            "like": r.get("like", 0),
            "collect": r.get("collect", 0),
            "comment": r.get("comment", 0),
            "ts": float(r["ts"]),
        }
        w = implicit_weight(evt, now=now)
        if w > 0:
            by_u[str(r["user_id"])].append((str(r["item_id"]), w))
    return by_u

def compute_item_sim(by_user: Dict[str, List[Tuple[str, float]]], alpha: float = ICF_ALPHA) -> Dict[str, Dict[str, float]]:
    co = defaultdict(lambda: defaultdict(float))
    norm = defaultdict(float)
    for _, items in by_user.items():
        for i1, w1 in items:
            norm[i1] += w1 ** 2
            for i2, w2 in items:
                if i1 == i2:
                    continue
                co[i1][i2] += (w1 * w2) ** alpha
    sim = {}
    for i1, nbrs in co.items():
        sim[i1] = {}
        for i2, c in nbrs.items():
            sim[i1][i2] = c / (math.sqrt(norm[i1]) * math.sqrt(norm[i2]) + 1e-12)
    return sim

def recommend_itemcf(user_id: str, by_user: Dict[str, List[Tuple[str, float]]], item_sim: Dict[str, Dict[str, float]], K: int = ICF_RECALL):
    seen = {i for i, _ in by_user.get(user_id, [])}
    scores = defaultdict(float)
    for i, w in by_user.get(user_id, []):
        for j, s in item_sim.get(i, {}).items():
            if j in seen:
                continue
            scores[j] += w * s
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K]

# =========================
# ALS
# =========================
def index_users_items(df_users: List[str], df_items: List[str]):
    u2i = {u: idx for idx, u in enumerate(df_users)}
    it2i = {it: idx for idx, it in enumerate(df_items)}
    i2it = {idx: it for it, idx in it2i.items()}
    return u2i, it2i, i2it

def build_sparse(df: pd.DataFrame, u2i: Dict[str, int], it2i: Dict[str, int], now: float) -> coo_matrix:
    rows, cols, data = [], [], []
    for _, r in df.iterrows():
        u, it = str(r["user_id"]), str(r["item_id"])
        if u not in u2i or it not in it2i:
            continue
        evt = {
            "click": r.get("click", 0),
            "finish": r.get("finish", 0),
            "like": r.get("like", 0),
            "collect": r.get("collect", 0),
            "comment": r.get("comment", 0),
            "ts": float(r["ts"]),
        }
        w = implicit_weight(evt, now=now)
        if w <= 0:
            continue
        rows.append(it2i[it])
        cols.append(u2i[u])
        data.append(w)
    return coo_matrix((data, (rows, cols)), shape=(len(it2i), len(u2i)))

def train_als(sparse_items_users: coo_matrix):
    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS, regularization=ALS_REG, iterations=ALS_ITERS
    )
    model.fit((sparse_items_users * ALS_ALPHA).astype("double"))
    return model

def recommend_als(model, S: coo_matrix, user_id: str, u2i: Dict[str, int], idx2item: Dict[int, str], N: int = ALS_RECALL):
    uid = u2i.get(user_id)
    if uid is None:
        return []
    recs = model.recommend(userid=uid, user_items=S.T.tocsr(), N=N, recalculate_user=True)
    return [(idx2item[i], float(s)) for i, s in recs]

# =========================
# 重排（新鲜度 + MMR）
# =========================
def mmr(cands: Dict[str, float], topN: int = TOPK_RETURN, lambda_: float = MMR_LAMBDA):
    selected = []
    remain = dict(cands)

    def cat(i: str) -> str:
        return item_meta[i]["category"]

    def sim(a: str, b: str) -> float:
        return 0.2 if cat(a) == cat(b) else 0.0

    for _ in range(topN):
        best, best_sc = None, -1e9
        for i, s in remain.items():
            div = 0.0 if not selected else max(sim(i, j) for j, _ in selected)
            sc = lambda_ * s - (1 - lambda_) * div
            if sc > best_sc:
                best, best_sc = (i, s), sc
        if not best:
            break
        selected.append(best)
        del remain[best[0]]
    return selected

# =========================
# 训练 & 主推荐
# =========================
def train_all(items_csv: str, interactions_csv: str):
    global item_meta, interactions_df, by_user, item_sim, user2idx, item2idx, idx2item, S, als_model, now_ts
    print("[TRAIN] loading data ...")
    item_meta = load_items(items_csv)
    interactions_df = load_interactions(interactions_csv)
    now_ts = datetime.now(timezone.utc).timestamp()

    users = sorted(set(interactions_df["user_id"].astype(str).tolist()))
    items = sorted(set(item_meta.keys()) | set(interactions_df["item_id"].astype(str).tolist()))
    user2idx, item2idx, idx2item = index_users_items(users, items)

    print("[TRAIN] build user history ...")
    by_user = build_user_item_weights(interactions_df, now=now_ts)

    print("[TRAIN] compute item-item similarity (ItemCF) ...")
    item_sim = compute_item_sim(by_user, alpha=ICF_ALPHA)

    print("[TRAIN] build sparse matrix & train ALS ...")
    S = build_sparse(interactions_df, user2idx, item2idx, now=now_ts)
    als_model = train_als(S)
    print("[TRAIN] done.")

def recommend_for_user(user_id: str, K: int = TOPK_RETURN):
    global now_ts
    now_ts = datetime.now(timezone.utc).timestamp()

    icf = dict(recommend_itemcf(user_id, by_user, item_sim, K=ICF_RECALL))
    als_rec = dict(recommend_als(als_model, S, user_id, user2idx, idx2item, N=ALS_RECALL))

    scores = defaultdict(float)
    for i, s in icf.items():
        scores[i] += MERGE_W_ICF * s
    for i, s in als_rec.items():
        scores[i] += MERGE_W_ALS * s

    seen = {i for i, _ in by_user.get(user_id, [])}
    cands = {}
    for i, s in scores.items():
        if i in seen:
            continue
        meta = item_meta.get(i)
        if not meta:
            continue
        if too_old(meta["publish_ts"], now_ts, MAX_AGE_DAYS):
            continue
        cands[i] = freshness_boost(s, meta["publish_ts"], now_ts)

    return mmr(cands, topN=K, lambda_=MMR_LAMBDA)

# =========================
# Lifespan（替代 on_event）
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] startup: training models...")
    train_all("data/items.csv", "data/interactions.csv")
    yield
    print("[LIFESPAN] shutdown: cleanup...")

app = FastAPI(title="News CF Recommender", lifespan=lifespan)

# =========================
# 路由
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "now": datetime.now(timezone.utc).isoformat()}

@app.get("/recommend")
def recommend_api(user_id: str = Query(...), k: int = Query(TOPK_RETURN, ge=1, le=200)):
    recs = recommend_for_user(user_id, K=k)
    out = [
        {
            "item_id": i,
            "score": float(s),
            "publish_ts": item_meta[i]["publish_ts"],
            "category": item_meta[i]["category"],
            "source": item_meta[i]["source"],
        }
        for i, s in recs
    ]
    return {"user_id": user_id, "count": len(out), "items": out}

@app.get("/retrain")
def retrain_api():
    train_all("data/items.csv", "data/interactions.csv")
    return {"ok": True}
