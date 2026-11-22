#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_align.py — المشغّل الكامل للمحاذاة بعد إعداد التضمينات.

أعيد بناء هذا الملف بعد جلسة Genspark (كان في بيئة العمل الأصلية ولم يُرفع
ضمن ortho_align_all.zip، فكان `cli_app.py align` يفشل باستيراده).

الخطوات (تظهر سجلًا واضحًا):
  1) تضمين كل الجمل بـLaBSE إن لم توجد en_emb.npy / ar_emb.npy (GPU موصى به)
  2) بناء عينات الفصول (embed_samples) → sample_meta.json
  3) الاسترجاع الدلالي الكامل: أقرب جملة عربية لكل جملة إنجليزية + margin
     → labse_retrieval.csv  [chapter,en,ar,confidence,margin]
  4) إقران مناطق الفصول (pair_regions.py) → regions.pkl
  5) المحاذاة DP بالمراسي (final_align.py) → aligned_pairs.csv

الاستخدام:
  python3 run_align.py [workdir] [outdir]     # افتراضي: . .
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _encode_missing(work: Path, sents):
    """تضمين الجمل الغائبة بـLaBSE (يتطلب torch + sentence-transformers)."""
    missing = [t for t in ("en", "ar") if not (work / f"{t}_emb.npy").exists()]
    if not missing:
        return
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        sys.exit(f"تغيب التضمينات: {missing} ويغيب النموذج: {e}\n"
                 f"ثبّت: pip install sentence-transformers torch")
    model = SentenceTransformer("sentence-transformers/LaBSE")
    model.max_seq_length = 128
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    for tag in ("en", "ar"):
        out_path = work / f"{tag}_emb.npy"
        if out_path.exists():
            continue
        texts = [r["text"] for r in sents[tag]]
        order = np.argsort([len(t) for t in texts])
        print(f"تضمين {tag}: {len(texts)} جملة ...", flush=True)
        e = model.encode([texts[i] for i in order], batch_size=96,
                         normalize_embeddings=True, show_progress_bar=True)
        np.save(out_path, e[np.argsort(order)].astype(np.float32))


def _retrieval_csv(work: Path, out: Path, chunk: int = 512):
    """أقرب جملة عربية لكل جملة إنجليزية (cosine) + margin عن الثاني."""
    sents = json.load(open(work / "sents.json", encoding="utf8"))
    E = np.load(work / "en_emb.npy")
    A = np.load(work / "ar_emb.npy")
    import csv
    t0 = time.time()
    rows = []
    for start in range(0, len(E), chunk):
        Ec = E[start:start + chunk]
        S = Ec @ A.T                                   # chunk × NAR
        top2_idx = np.argpartition(-S, 1, axis=1)[:, :2]
        top2_val = np.take_along_axis(S, top2_idx, axis=1)
        order = np.argsort(-top2_val, axis=1)
        best_i = top2_idx[np.arange(len(Ec)), order[:, 0]]
        best_v = top2_val[np.arange(len(Ec)), order[:, 0]]
        second_v = top2_val[np.arange(len(Ec)), order[:, 1]]
        for k in range(len(Ec)):
            i_en, i_ar = start + k, int(best_i[k])
            rows.append({
                "chapter": sents["en"][i_en]["chapter"].replace("_2026", ""),
                "en": sents["en"][i_en]["text"],
                "ar": sents["ar"][i_ar]["text"],
                "confidence": round(float(best_v[k]), 3),
                "margin": round(float(best_v[k] - second_v[k]), 3),
            })
        done = min(start + chunk, len(E))
        print(f"  retrieval {done}/{len(E)} ({time.time()-t0:.0f}s)", flush=True)
    path = out / "labse_retrieval.csv"
    with open(path, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chapter", "en", "ar", "confidence", "margin"])
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    b75 = sum(1 for r in rows if r["confidence"] >= 0.75)
    b75m5 = sum(1 for r in rows if r["confidence"] >= 0.75 and r["margin"] >= 0.05)
    print(f"WROTE {path} rows={n} | conf>=0.75: {b75} | +margin>=0.05: {b75m5}", flush=True)


def main(workdir=".", outdir="."):
    work, out = Path(workdir), Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sents = json.load(open(work / "sents.json", encoding="utf8"))
    print(f"sents: en={len(sents['en'])} ar={len(sents['ar'])}", flush=True)

    print("== 1/5 التضمينات ==", flush=True)
    _encode_missing(work, sents)

    print("== 2/5 عينات الفصول ==", flush=True)
    import embed_samples
    embed_samples.main(str(work))

    print("== 3/5 الاسترجاع الدلالي الكامل ==", flush=True)
    _retrieval_csv(work, out)

    print("== 4/5 إقران مناطق الفصول ==", flush=True)
    pkg = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(pkg / "pair_regions.py")], cwd=str(work), check=True)

    print("== 5/5 المحاذاة DP بالمراسي ==", flush=True)
    # pair_regions يكتب regions.pkl بينما final_align يقرأ chapter_pairs_emb.pkl — ربط الاسمين
    if not (work / "chapter_pairs_emb.pkl").exists() and (work / "regions.pkl").exists():
        shutil.copy(work / "regions.pkl", work / "chapter_pairs_emb.pkl")
    subprocess.run([sys.executable, str(pkg / "final_align.py")], cwd=str(work), check=True)

    dp_src = work / "aligned_pairs.csv"
    if dp_src.exists():
        shutil.copy(dp_src, out / "aligned_pairs.csv")
        print("OK:", str((out / "aligned_pairs.csv").resolve()), flush=True)
    print("OK:", str((out / "labse_retrieval.csv").resolve()), flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:3] or [".", "."]))
