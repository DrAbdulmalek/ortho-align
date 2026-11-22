#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final stage: use Hungarian-assigned (chapter, block) pairs from LaBSE sample
embeddings, expand to full regions (midpoints between adjacent blocks), run
anchor-based DP sentence alignment per pair, write aligned_pairs.csv."""
import json, csv, pickle, time
import numpy as np
from dp0 import dp_align, to_conf   # anchor DP (dp0 = dp_align module without emb dependency)

sents = json.load(open("sents.json", encoding="utf8"))
pairs = pickle.load(open("chapter_pairs_emb.pkl", "rb"))
NAR = len(sents["ar"])

# --- expand blocks to full-coverage regions using midpoints between block spans
pairs_sorted = sorted(pairs, key=lambda p: p[1])
regions = []
for k, (ch, lo, hi, s) in enumerate(pairs_sorted):
    lo2 = 0 if k == 0 else (pairs_sorted[k-1][2] + hi) // 2
    hi2 = NAR if k == len(pairs_sorted) - 1 else (hi + pairs_sorted[k+1][1]) // 2
    regions.append((ch, lo2, hi2, s))

rows = []
t0 = time.time()
for pi, (ch, lo, hi, csim) in enumerate(regions):
    ix = [i for i, r in enumerate(sents["en"]) if r["chapter"] == ch]
    en_block = [sents["en"][i]["text"] for i in ix]
    ar_block = [r["text"] for r in sents["ar"][lo:hi]]
    links = dp_align(en_block, ar_block)
    kept = 0
    for ii, jj, sc in links:
        if ii and jj and sc is not None and sc >= 0.55:
            kept += 1
            rows.append({
                "chapter": ch.replace("_2026", ""),
                "en": " ".join(en_block[a] for a in ii),
                "ar": " ".join(ar_block[b] for b in jj),
                "link_type": f"{len(ii)}:{len(jj)}",
                "confidence": to_conf(sc),
                "chapter_pair_sim": round(csim, 3),
            })
    print(f"[{pi+1}/{len(regions)}] {ch}: en={len(ix)} ar={hi-lo} kept={kept} ({time.time()-t0:.0f}s)", flush=True)

with open("aligned_pairs.csv", "w", encoding="utf8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["chapter", "en", "ar", "link_type", "confidence", "chapter_pair_sim"])
    w.writeheader(); w.writerows(rows)
print(f"WROTE aligned_pairs.csv rows={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
