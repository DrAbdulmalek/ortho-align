#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""إعداد التضمينات النموذجية (sample embeddings) لإقران الفصول.

أعيد بناء هذا الملف بعد جلسة Genspark — كان موجودًا في بيئة العمل الأصلية
ولم يُرفع ضمن الحزمة. يشغّله run_align.py تلقائيًا؛ لا حاجة لاستدعائه يدويًا
إلا لإعادة بناء العينات.

المخرجات في مجلد العمل:
  en_sample_emb.npy  — تضمينات الإنجليزية (كل الجمل، مراكز الفصول دقيقة)
  ar_sample_emb.npy  — تضمينات العربية (كل جملة رابعة: 0,4,8,...)
  sample_meta.json   — {"en_idx": [...], "ar_idx": [...]} فهرس كل صف إلى جملته
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(workdir="."):
    work = Path(workdir)
    sents = json.load(open(work / "sents.json", encoding="utf8"))
    en_emb = np.load(work / "en_emb.npy")
    ar_emb = np.load(work / "ar_emb.npy")

    en_idx = list(range(len(sents["en"])))
    ar_idx = list(range(0, len(sents["ar"]), 4))   # عينة كل 4 جمل عربية (موثق في README)

    np.save(work / "en_sample_emb.npy", en_emb[en_idx].astype(np.float32))
    np.save(work / "ar_sample_emb.npy", ar_emb[ar_idx].astype(np.float32))
    json.dump({"en_idx": en_idx, "ar_idx": ar_idx},
              open(work / "sample_meta.json", "w", encoding="utf8"))
    print(f"samples: en={len(en_idx)} ar={len(ar_idx)} (من {len(sents['ar'])})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
