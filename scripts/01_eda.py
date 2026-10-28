#!/usr/bin/env python3
"""Stage 1: MaleCNS 连接组统计探索 —— 提取"发育规律"的原始素材"""
import pandas as pd
import numpy as np
import json
import os

D = "/tmp/flydata/"
OUT = "/tmp/flydata/stats/"
os.makedirs(OUT, exist_ok=True)

print("加载注释表...")
ann = pd.read_feather(D + "body-annotations.feather")
print("annotations:", ann.shape)
print("columns:", ann.columns.tolist())

print("加载连接表...")
edges = pd.read_feather(D + "connectome-weights.feather")
print("edges:", edges.shape)
print("columns:", edges.columns.tolist())
print(edges.head())

# 保存 schema 供后续使用
with open(OUT + "schema.json", "w") as f:
    json.dump({"ann_cols": ann.columns.tolist(), "edge_cols": edges.columns.tolist(),
               "n_ann": len(ann), "n_edges": len(edges)}, f, ensure_ascii=False, indent=2)
print("DONE")
