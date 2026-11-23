#!/usr/bin/env python3
"""اختبار دخاني شامل لحزمة ortho-align (بلا نموذج LaBSE — تضمينات صناعية).

يبني كتالوجًا صناعيًا: 3 فصول إنجليزية × 60 جملة، كتاب عربي بترتيب فصول معكوس،
تضمينات صناعية حيث كل جملة عربية = نسخة مضوضاة من مقابلها الإنجليزي،
ثم يشغل السلسلة كاملة: prepare → run_align (استرجاع + مناطق + DP) ويتحقق
من الدقة مقابل الحقيقة الأرضية المعروفة.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
SMOKE = PKG / ".smoke_tmp"
sys.path.insert(0, str(PKG))

# ── 1) بناء الكتالوج الصناعي ────────────────────────────────────────────
if SMOKE.exists():
    shutil.rmtree(SMOKE)
(SMOKE / "work2" / "en").mkdir(parents=True)
(SMOKE / "work2" / "ar").mkdir(parents=True)
(SMOKE / "out").mkdir()

EN_CH = ["Orthobullets_A_2026", "Orthobullets_B_2026", "Orthobullets_C_2026"]
LET = {c: c[13] for c in EN_CH}                    # الحرف A/B/C من الاسم
AR_ORDER = [EN_CH[2], EN_CH[0], EN_CH[1]]          # ترتيب الفصول العربي معكوس عمدًا
N = 600

en_lines, en_global = [], []                        # en_global: فصل كل جملة إنجليزية
for c in EN_CH:
    for i in range(N):
        en_lines.append(f"Procedure T{LET[c]}{i:03d} shows {i + 10}% success and 3.{i} mm fixation.")
        en_global.append(c)
    # كل فصل في ملفه (chapter = اسم الملف كما في الكتاب الحقيقي المقسم بالأقسام)
    (SMOKE / "work2" / "en" / f"{c}.txt").write_text(
        "\n".join(en_lines[-N:]), encoding="utf8")

ar_lines, ar_truth = [], []                         # ar_truth: فهرس الجملة الإنجليزية المقابلة
for c in AR_ORDER:
    base = EN_CH.index(c) * N
    for i in range(N):
        ar_lines.append(f"إجراء فصل {LET[c]} مرقم {i:03d} بنسبة نجاح {i + 10} بالمئة وثبت 3.{i} مم تكنيك T{LET[c]}{i:03d}.")
        ar_truth.append(base + i)
(SMOKE / "work2" / "ar" / "Orthobullets_ar_2021.txt").write_text("\n".join(ar_lines), encoding="utf8")

# ── 2) التقطيع (prepare) ───────────────────────────────────────────────
import prepare
prepare.main(str(SMOKE / "work2"), str(SMOKE / "work2" / "sents.json"))
sents = json.load(open(SMOKE / "work2" / "sents.json", encoding="utf8"))
print(f"EN={len(sents['en'])} AR={len(sents['ar'])} (متوقع {len(en_lines)}/{len(ar_lines)})")

# ── 3) تضمينات صناعية: مقابل العربي = نسخة مضوضاة من الإنجليزي ─────────
rng = np.random.default_rng(7)
DIM = 96
base = rng.standard_normal((len(en_lines), DIM))
base /= np.linalg.norm(base, axis=1, keepdims=True)
en_emb = base + rng.standard_normal(base.shape) * 0.02
ar_emb = np.stack([base[ar_truth[k]] + rng.standard_normal(DIM) * 0.02 for k in range(len(ar_lines))])
for v in (en_emb, ar_emb):
    v /= np.linalg.norm(v, axis=1, keepdims=True)
np.save(SMOKE / "work2" / "en_emb.npy", en_emb.astype(np.float32))
np.save(SMOKE / "work2" / "ar_emb.npy", ar_emb.astype(np.float32))

# ── 4) السلسلة الكاملة المصلحة ─────────────────────────────────────────
import run_align
run_align.main(str(SMOKE / "work2"), str(SMOKE / "out"))

# ── 5) التحقق مقابل الحقيقة الأرضية ────────────────────────────────────
import csv
rows = list(csv.DictReader(open(SMOKE / "out" / "labse_retrieval.csv", encoding="utf8")))
print(f"\nlabse_retrieval.csv rows={len(rows)} (متوقع {len(en_lines)})")

# إعادة بناء فهرس العربي كما في sents.json (بعد التقطيع): الجملة العربية رقمها = ترتيبها
ar_texts = [r["text"] for r in sents["ar"]]
def ar_index_of(k):
    marker = f"مرقم {k:03d}"
    for j, t in enumerate(ar_texts):
        if marker in t:
            return j
    return None

hit = miss = 0
for r in rows:
    # استخرج رقم الجملة من النص الإنجليزي (T{ch}{idx:03d})
    import re
    m = re.search(r"T([ABC])(\d{3})", r["en"])
    k = int(m.group(2))
    expected = ar_index_of(k)
    if expected is None:
        continue
    # تحقق أن النص العربي الأعلى ثقة يحمل نفس الرقم
    ok = f"مرقم {k:03d}" in r["ar"]
    hit += ok
    miss += (not ok)
print(f"استرجاع صحيح: {hit}/{hit+miss} ({100*hit/max(1,hit+miss):.1f}%)")

dp = list(csv.DictReader(open(SMOKE / "out" / "aligned_pairs.csv", encoding="utf8")))
print(f"aligned_pairs.csv rows={len(dp)} | chapters={sorted({r['chapter'] for r in dp})}")
print(f"link_type distribution:", {lt: sum(1 for r in dp if r['link_type'] == lt) for lt in {r['link_type'] for r in dp}})

ok_rows = (len(rows) == len(en_lines)) and (len(dp) > 100) and hit / max(1, hit + miss) >= 0.9
print("\n" + ("✅ SMOKE PASS — السلسلة المصلحة تعمل من البداية للنهاية" if ok_rows else "❌ SMOKE FAIL"))
sys.exit(0 if ok_rows else 1)
