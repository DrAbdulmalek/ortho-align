#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: normalization + sentence segmentation (EN & AR)."""
import re, json, unicodedata, glob, os
from pathlib import Path

RTL = "\u200f\u200e\u202a\u202b\u202c\u202d\u202e\ufeff"
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def norm(t):
    t = unicodedata.normalize("NFKC", t)
    for ch in RTL: t = t.replace(ch, "")
    t = t.replace("\u0640", "")          # tatweel
    t = t.translate(AR_DIGITS)
    t = re.sub(r"[\u25a1\u25aa-\u25ae\u25af\u2217\u2022\u00b7\u25cf\u25cb]", " ", t)
    return t

EN_ABBREV = re.compile(r"(?i)\b(?:e\.g|i\.e|etc|et al|dr|mr|mrs|ms|prof|sr|jr|vol|fig|figs|no|nos|pp|cf|vs|approx|ref|cc|ml|mg|mcg|kg|cm|mm|hr|min|sec|bpm|cbc|ct|mri|pet|ecg|ekg|emg|ota|dco|qid|bid|tid|po|iv|sq|prn|approx|al)\.(?=\s)")

def protect(text, regex, store):
    return regex.sub(lambda m: (store.append(m.group(0)) or f"\x00{len(store)-1}\x00"), text)

def unprotect(s, store):
    s = re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], s)
    return s.replace("\x00D\x00", ".")

def split_en(text):
    store = []
    t = protect(text, EN_ABBREV, store)
    t = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\x00D\x00" + m.group(2), t)
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    return [unprotect(p, store).strip() for p in parts]

def split_ar(text):
    parts = re.split(r"(?<=[.؟!])\s+|\n+", text)
    return [p.strip() for p in parts]

JUNK_EN = re.compile(r"^(created by:|created by|updated:?|table of contents|https?://|www\.)", re.I)
JUNK_AR = re.compile(r"^(اإلجابات|األجوبة|الرجوع|جميع الحقوق|http)", re.I)

def is_junk(s, ar=False):
    st = s.strip()
    if len(st.split()) < 3: return True
    if re.fullmatch(r"[\d\W\s]+", st): return True
    if "http" in st.lower(): return True
    if re.fullmatch(r"[\x0c\s]*", st): return True
    if ar: return bool(JUNK_AR.match(st))
    return bool(JUNK_EN.match(st))

def main(book_dir, out):
    recs = {"en": [], "ar": []}
    for f in sorted(glob.glob(f"{book_dir}/en/*.txt")):
        text = norm(Path(f).read_text(encoding="utf8", errors="ignore"))
        ch = Path(f).stem
        for s in split_en(text):
            s = re.sub(r"\s+", " ", s).strip().strip("\x0c").strip()
            if s and not is_junk(s):
                recs["en"].append({"chapter": ch, "text": s})
    for f in sorted(glob.glob(f"{book_dir}/ar/*.txt")):
        text = norm(Path(f).read_text(encoding="utf8", errors="ignore"))
        for s in split_ar(text):
            s = re.sub(r"\s+", " ", s).strip().strip("\x0c").strip()
            if s and not is_junk(s, ar=True):
                recs["ar"].append({"chapter": Path(f).stem, "text": s})
    json.dump(recs, open(out, "w", encoding="utf8"), ensure_ascii=False)
    print("EN sentences:", len(recs["en"]), "| AR sentences:", len(recs["ar"]))
    print("EN chars:", sum(len(r['text']) for r in recs['en']), "AR chars:", sum(len(r['text']) for r in recs['ar']))

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
