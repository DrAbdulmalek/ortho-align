#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-coverage chapter regions from LaBSE sample votes.
Every AR sentence gets the chapter of its smoothed neighborhood => contiguous
regions that tile the whole Arabic book (no gaps)."""
import json, pickle
import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

sents = json.load(open("sents.json", encoding="utf8"))
meta = json.load(open("sample_meta.json"))
en_recs, ar_recs = sents["en"], sents["ar"]
E = np.load("en_sample_emb.npy"); A = np.load("ar_sample_emb.npy")
en_idx, ar_idx = meta["en_idx"], meta["ar_idx"]
en_chapters = sorted({r["chapter"] for r in en_recs})
NAR = len(ar_recs)
cen = {}
for c in en_chapters:
    ix = [k for k, i in enumerate(en_idx) if en_recs[i]["chapter"] == c]
    v = E[ix].mean(0); cen[c] = v / np.linalg.norm(v)
K = len(en_chapters)
S = np.stack([A @ cen[c] for c in en_chapters], axis=1)
best_ch = S.argmax(1)
full_best = np.full(NAR, -1, dtype=np.int32)
for k, i in enumerate(ar_idx):
    full_best[i] = best_ch[k]
# windowed majority (window over voted sentences only)
W = 120
cs = np.cumsum(np.stack([(full_best == k).astype(np.int32) for k in range(K)]), axis=1)
cnt_all = np.cumsum((full_best >= 0).astype(np.int32))
sm = np.full(NAR, -1, dtype=np.int32)
for i in range(NAR):
    lo, hi = max(0, i - W), min(NAR, i + W)
    cnt = cs[:, hi - 1] - (cs[:, lo - 1] if lo > 0 else 0)
    tot = cnt_all[hi - 1] - (cnt_all[lo - 1] if lo > 0 else 0)
    if tot >= 8:
        sm[i] = int(cnt.argmax())
# fill unvoted gaps with nearest voted label
voted = np.where(sm >= 0)[0]
if len(voted):
    idx = np.searchsorted(voted, np.arange(NAR))
    idx = np.clip(idx, 1, len(voted) - 1)
    left = voted[np.clip(idx - 1, 0, len(voted) - 1)]
    right = voted[np.clip(idx, 0, len(voted) - 1)]
    d = np.abs(np.arange(NAR) - left); d2 = np.abs(right - np.arange(NAR))
    sm = np.where(sm >= 0, sm, np.where(d <= d2, sm[left], sm[right]))
# contiguous regions with min size 60
MINB = 60
bounds = [0] + [i for i in range(1, NAR) if sm[i] != sm[i - 1]] + [NAR]
merged = []
for b in range(len(bounds) - 1):
    lo, hi = bounds[b], bounds[b + 1]
    if merged and hi - lo < MINB: merged[-1] = (merged[-1][0], hi, merged[-1][2])
    else: merged.append((lo, hi, int(sm[lo]) if sm[lo] >= 0 else -1))
# block-vs-chapter Hungarian on mean sample similarity inside region
sim_region = np.zeros((len(merged), K))
for bi, (lo, hi, lab) in enumerate(merged):
    ks = [k for k, i in enumerate(ar_idx) if lo <= i < hi]
    if ks: sim_region[bi] = S[ks].mean(0)
ri, ci = linear_sum_assignment(-sim_region)
regions = []
for r, c in zip(ri, ci):
    lo, hi, lab = merged[r]
    regions.append((en_chapters[c], lo, hi, float(sim_region[r, c])))
regions.sort(key=lambda x: x[1])
print("regions:", len(regions))
for ch, lo, hi, s in regions:
    print(f"  {s:.3f} {ch[12:].replace('_2026',''):<22} AR[{lo}:{hi}] ({hi-lo})")
pickle.dump(regions, open("regions.pkl", "wb"))
