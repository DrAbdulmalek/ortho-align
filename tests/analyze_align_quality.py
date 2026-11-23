#!/usr/bin/env python3
"""تحليل جودة مخرجات المحاذاة المُسلَّمة من جلسة Genspark (يعمل بـ -I)."""
import csv
import re
from collections import Counter

import sys
from pathlib import Path as _P
BASE = sys.argv[1] if len(sys.argv) > 1 else str(_P(__file__).resolve().parent.parent)
if not BASE.endswith("/"):
    BASE += "/"

def load(name):
    with open(BASE + name, encoding="utf8") as f:
        return list(csv.DictReader(f))

lab = load("labse_retrieval.csv")
dp = load("aligned_pairs.csv")

print("=== labse_retrieval.csv ===")
print("rows:", len(lab), "| columns:", list(lab[0].keys()))
conf = [float(r["confidence"]) for r in lab]
marg = [float(r["margin"]) for r in lab]
n = len(lab)
b75 = sum(1 for c in conf if c >= 0.75)
b75m5 = sum(1 for c, m in zip(conf, marg) if c >= 0.75 and m >= 0.05)
b65 = sum(1 for c in conf if c >= 0.65)
b60 = sum(1 for c in conf if c >= 0.60)
print(f"conf>=0.75: {b75} ({100*b75/n:.1f}%) | +margin>=0.05: {b75m5} ({100*b75m5/n:.1f}%)")
print(f"conf>=0.65: {b65} ({100*b65/n:.1f}%) | conf>=0.60: {b60} ({100*b60/n:.1f}%)")
print("conf quantiles: max=%.3f p50=%.3f p90=%.3f min=%.3f" % (
    max(conf), sorted(conf)[n//2], sorted(conf)[int(n*0.9)], min(conf)))
ch = Counter(r["chapter"] for r in lab)
print("chapters:", len(ch), "| top:", ch.most_common(3), "| empty-ar:", sum(1 for r in lab if not r["ar"].strip()))
en_dup = [k for k, v in Counter(r["en"] for r in lab).items() if v > 1]
print("duplicate EN sentences:", len(en_dup))

# كسر النص العربي: حروف مفردة معزولة + محارف الاستبدال U+FFFD
def ar_garble(s):
    toks = s.split()
    if not toks: return True
    single = sum(1 for t in toks if len(t) == 1 and "\u0600" <= t <= "\u06FF")
    if "\ufffd" in s: return True
    return single / len(toks) > 0.25

gar = sum(1 for r in lab if ar_garble(r["ar"]))
print(f"AR garbled rows (isolated letters / U+FFFD): {gar} ({100*gar/n:.1f}%)")

print()
print("=== aligned_pairs.csv ===")
print("rows:", len(dp), "| columns:", list(dp[0].keys()))
dconf = [float(r["confidence"]) for r in dp]
print(f"conf>=0.55: {sum(1 for c in dconf if c>=0.55)} ({100*sum(1 for c in dconf if c>=0.55)/len(dp):.1f}%)")
print("link_type:", Counter(r.get("link_type", "?") for r in dp).most_common(6))
dch = Counter(r["chapter"] for r in dp)
print("chapters:", len(dch), "| top:", dch.most_common(3))

print()
print("=== spotcheck50.txt — فحص بشري على عينة QC ===")
txt = open(BASE + "qc_retrieval.txt", encoding="utf8").read()
blocks = re.findall(r"--- \d+ \[([^\]]+)\]\nEN: (.*?)\nAR: (.*?)(?=\n--- |\Z)", txt, re.S)
print("QC rows in file:", len(blocks))
