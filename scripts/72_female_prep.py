#!/usr/bin/env python3
"""72_female_prep.py — FlyWire FAFB v783 (female) census builder.

Mirrors 32_census_prep.py for the male, but builds the node table directly in
type-grouped order (avoiding the male compact-order label erratum, §22.2).

Inputs (data_female/):
  consolidated_cell_types.csv.gz   root_id,primary_type,...
  coordinates.csv.gz               root_id,position([x y z] nm),...
  connections_princeton_no_threshold.csv.gz  pre_root_id,post_root_id,neuropil,syn_count,nt_type

Outputs (data_female/):
  node_ids.i64      sorted root_ids of typed nodes, order = (type_idx, root_id)
  node_type.i32     type index per node
  type_names.npy    type strings
  xyz.f32           soma position (nm -> um), same order
  e_pre/e_post/e_w  aggregated typed-typed edges (syn_count sum)
"""
import numpy as np, pandas as pd, os, sys, time, resource

D = "/mnt/agents/output/果蝇生成器/data_female/"
CONN = os.environ.get("CONN", D + "connections_princeton_no_threshold.csv.gz")
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

# ---------- nodes ----------
ct = pd.read_csv(D + "consolidated_cell_types.csv.gz")
ct = ct[ct.primary_type.notna()].copy()
log("typed rows:", len(ct))

coo = pd.read_csv(D + "coordinates.csv.gz")
pos = coo.position.str.strip("[]").str.split(expand=True).astype(np.float64).values
coo_xyz = {}
# mean position per root (multiple rows per root possible)
rid = coo.root_id.values
order = np.argsort(rid, kind="stable")
rid_s = rid[order]; pos_s = pos[order]
uq, st = np.unique(rid_s, return_index=True)
st = np.append(st, len(rid_s))
xyz_map = np.zeros((len(uq), 3))
for i in range(len(uq)):
    xyz_map[i] = pos_s[st[i]:st[i+1]].mean(0)
del pos, pos_s, rid, rid_s, order
log("coords:", len(uq), "rss", rss(), "MB")

# merge coords onto typed neurons
ct = ct.merge(pd.DataFrame({"root_id": uq, "x": xyz_map[:,0], "y": xyz_map[:,1], "z": xyz_map[:,2]}),
              on="root_id", how="left")
log("typed with coords:", ct.x.notna().sum(), "/", len(ct))

types = np.sort(ct.primary_type.unique())
tmap = {t: i for i, t in enumerate(types)}
ct["tidx"] = ct.primary_type.map(tmap).astype(np.int32)

# node order: (type, root_id) — genome order, offsets are contiguous blocks
ct = ct.sort_values(["tidx", "root_id"], kind="stable").reset_index(drop=True)
N = len(ct)
node_ids = ct.root_id.values.astype(np.int64)
node_type = ct.tidx.values.astype(np.int32)
xyz = ct[["x","y","z"]].values.astype(np.float32) / 1000.0  # nm -> um
xyz[np.isnan(xyz)] = np.nan
node_ids.tofile(D + "node_ids.i64")
node_type.tofile(D + "node_type.i32")
np.save(D + "type_names.npy", types.astype(object))
xyz.tofile(D + "xyz.f32")
tc = np.bincount(node_type, minlength=len(types))
offsets = np.zeros(len(types)+1, np.int64); np.cumsum(tc, out=offsets[1:])
offsets.tofile(D + "offsets.i64")
log(f"nodes={N}, T={len(types)}, rss={rss()}MB")

# ---------- edges ----------
id2idx = node_ids  # sorted? node_ids sorted within type blocks but NOT globally.
# build a global sorted lookup
gorder = np.argsort(node_ids)
gids = node_ids[gorder]

def map_ids(arr):
    p = np.searchsorted(gids, arr)
    p = np.clip(p, 0, N-1)
    out = np.where(gids[p] == arr, gorder[p], -1)
    return out

chunks = []
cols = ["pre_root_id", "post_root_id", "syn_count"]
tot = 0
for i, ch in enumerate(pd.read_csv(CONN,
                                   usecols=cols, chunksize=4_000_000)):
    pi = map_ids(ch.pre_root_id.values)
    qi = map_ids(ch.post_root_id.values)
    ok = (pi >= 0) & (qi >= 0)
    key = pi[ok].astype(np.int64) * N + qi[ok].astype(np.int64)
    w = ch.syn_count.values[ok].astype(np.int64)
    chunks.append((key, w))
    tot += int(ok.sum())
    if i % 5 == 0:
        log(f"chunk {i}, kept so far {tot}, rss={rss()}MB")
    del ch, pi, qi, ok
key = np.concatenate([c[0] for c in chunks])
w = np.concatenate([c[1] for c in chunks])
chunks = None
log("edges typed-typed (pre-agg):", len(key), "rss", rss(), "MB")
o = np.argsort(key, kind="stable")
key = key[o]; w = w[o]
uqk, st = np.unique(key, return_index=True)
st = np.append(st, len(key))
ws = np.add.reduceat(w, st[:-1])
e_pre = (uqk // N).astype(np.int32)
e_post = (uqk % N).astype(np.int32)
e_w = ws.astype(np.int32)
log("unique pairs:", len(uqk), "rss", rss(), "MB")
e_pre.tofile(D + "e_pre.i32"); e_post.tofile(D + "e_post.i32"); e_w.tofile(D + "e_w.i32")
np.save(D + "meta.npy", {"N": N, "T": len(types), "E": len(uqk)}, allow_pickle=True)
log("PREP DONE. N=%d T=%d E=%d" % (N, len(types), len(uqk)))
