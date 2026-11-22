#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align.py — محاذاة الجمل بين كتاب Orthobullets الإنجليزي (2026) والعربي (2021).

الفكرة الجوهرية:
 الترجمة العربية لا تحفظ الجملة الإنجليزية، لكنها تحفظ «المراسي» (anchors):
 الأرقام والنِّسب، المصطلحات اللاتينية، أسماء التصنيفات والأدوية، أكواد الأسئلة
 (OBQ…) — وهي لغة-مستقلة فتصلح جسرًا بين النصين.

المرحلة 1 — إقران الفصول بالمحتوى (لأن ترتيب الفصول مختلف بين النسختين):
 لكل فصل إنجليزي نستخرج أندر مراسيه (أعلى وزن TF-IDF)، ثم نبحث مواقع هذه المراسي
 في النص العربي؛ فصلٌ يظهر نصابه في ناحية واحدة من العربي. نرتّب الفصول حسب
 الوسيط (median) لمواقع مراسيها ونقطع العربي بين الوسطاء — لا حاجة لفرض الترتيب.

المرحلة 2 — محاذاة داخل كل زوج فصول:
 برمجة ديناميكية بأسلوب Needleman–Wunsch (عائلة Gale–Church) بخمس عمليات:
 1:1 / 1:2 / 2:1 / 1:0 / 0:1. النقاط = تداخل المراسي (قوي) + نسبة الطول (ضعيف)،
 مع نطاق (band) حول القطر التناسبي لأن الترتيب داخل الفصل أحادي اللغة.

كل رابط يحمل confidence بين 0 و1 — الخوارزمية لا تدّعي دقة 100%، بل تُعلِّم
الصفوف غير المؤكدة بدل إخفائها.

الاستخدام:
  python3 align.py            # إقران الفصول + المحاذاة => aligned_pairs.csv
  python3 align.py sample 50  # فحص عشوائي لـ 50 صفا (طباعة للمراجعة)
"""
import json, csv, re, math, sys, time, random, statistics
import numpy as np
from collections import Counter, defaultdict

# ---------------------------------------------------------------- المراسي
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9\-']{1,}")
NUM = re.compile(r"\d+(?:[.,]\d+)*\s*%?")
STOP_LATIN = set("""table figure figures chapter updated summary created https www
page pages et al fig figs no nos vs see above below following use used using
key point points question questions answer answers choice choices concept tested""".split())

def anchors(s):
    """المراسي: أرقام + كلمات لاتينية (مع استبعاد كلمات الهيكل العامة)."""
    lat = {t.lower() for t in LATIN.findall(s)} - STOP_LATIN
    return lat | set(NUM.findall(s))

# ------------------------------------------------- المرحلة 1: إقران الفصول
def chapter_pairs(sents, top_k=400, min_hits=15, verbose=True):
    """إقران الفصول الإنجليزية بمناطقها في العربي عبر وسط مواقع المراسي النادرة."""
    en_recs, ar_recs = sents["en"], sents["ar"]
    en_chapters = sorted({r["chapter"] for r in en_recs})
    K = len(en_chapters)
    NAR = len(ar_recs)

    ch_tf, ar_df = {}, Counter()
    for c in en_chapters:
        tf = Counter()
        for r in en_recs:
            if r["chapter"] == c:
                tf.update(anchors(r["text"]))
        ch_tf[c] = tf
        for a in tf:
            ar_df[a] += 1
    idf = {a: math.log((K + 1) / (df + 0.5)) for a, df in ar_df.items()}

    # مواقع كل مرساة في النص العربي
    ar_pos = defaultdict(list)
    for i, r in enumerate(ar_recs):
        for a in anchors(r["text"]):
            ar_pos[a].append(i)

    med = {}
    for c in en_chapters:
        w = Counter()
        for a, t in ch_tf[c].items():
            w[a] = (1 + math.log(t)) * idf.get(a, 1.0)
        rare = [a for a, _ in w.most_common(top_k)]
        hits = []
        for a in rare:
            for p in ar_pos.get(a, ()):
                hits.append(p)
        if len(hits) >= min_hits:
            med[c] = (float(statistics.median(hits)), len(hits))
        else:
            med[c] = (None, len(hits))
        if verbose:
            print(f"  {c[12:].replace('_2026',''):<22} anchor-hits={med[c][1]:<6} median={med[c][0]}", flush=True)

    unmatched = [c for c in en_chapters if med[c][0] is None]
    matched = [c for c in en_chapters if med[c][0] is not None]
    matched.sort(key=lambda c: med[c][0])
    # الحدود: منتصف المسافة بين وسيط كل فصل والتالي
    bounds = [0]
    for a, b in zip(matched, matched[1:]):
        bounds.append(int((med[a][0] + med[b][0]) // 2))
    bounds.append(NAR)
    pairs = []
    for c, (lo, hi) in zip(matched, zip(bounds, bounds[1:])):
        pairs.append((c, lo, hi, med[c][1]))
    if verbose:
        for c in unmatched:
            print(f"  UNMATCHED (قلة المراسي): {c}", flush=True)
        print(f"أزواج فصول: {len(pairs)}/{K}", flush=True)
    return pairs, unmatched

# --------------------------------------- المرحلة 2: محاذاة DP داخل الزوج
GAP = -0.55
MU, SIG = 1.15, 0.45   # نسبة الطول المتوقعة عربي/إنجليزي (حروف)

def len_score(le, la):
    r = math.log((la / le) / MU)
    return max(-1.5, -0.5 * (r / SIG) ** 2)

def dp_align(en_s, ar_s, band_frac=0.06, min_band=250, verbose=False):
    n, m = len(en_s), len(ar_s)
    if n == 0 or m == 0:
        return []
    enA = [anchors(s) for s in en_s]; arA = [anchors(s) for s in ar_s]
    enN = [{a for a in x if a[0].isdigit()} for x in enA]
    arN = [{a for a in x if a[0].isdigit()} for x in arA]
    enL = [max(6, len(s)) for s in en_s]; arL = [max(6, len(s)) for s in ar_s]

    NEG = -1e9
    band = max(min_band, int(band_frac * max(n, m)))
    prev = [NEG] * (m + 1); prev[0] = 0.0
    prevprev = None
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + GAP
        bt[0, j] = 4

    for i in range(1, n + 1):
        cur = [NEG] * (m + 1)
        center = int(round(i * m / n))
        lo = max(1, center - band); hi = min(m, center + band)
        for j in range(1, lo):                       # خارج النطاق (يسار): فجوات
            cur[j] = cur[j - 1] + GAP
            bt[i, j] = 4
        eA, eN = enA[i - 1], enN[i - 1]
        e_lats = max(1, len(eA) - len(eN))
        ls_i = len_score(enL[i - 1], 10)  # placeholder not used directly
        for j in range(lo, hi + 1):
            best, code = NEG, 3
            v = prev[j] + GAP                        # 1:0
            if v > best: best, code = v, 3
            aA, aN = arA[j - 1], arN[j - 1]
            inter = eA & aA
            sc11 = None
            if inter:
                ratio = len(inter) / max(2.0, math.sqrt(max(1, len(eA) * len(aA))))
                num_hit = len(eN & aN) / max(1, len(eN))
                lat_hit = len((eA - eN) & (aA - aN)) / e_lats
                sc11 = 1.4 * ratio + 1.6 * num_hit + 1.0 * lat_hit + 0.8 * len_score(enL[i - 1], arL[j - 1])
                v = prev[j - 1] + sc11               # 1:1
                if v > best: best, code = v, 1
            v = cur[j - 1] + GAP                     # 0:1
            if v > best: best, code = v, 4
            if j >= 2:                               # 1:2
                aA2, aN2 = arA[j - 2], arN[j - 2]
                inter2 = eA & aA2
                if inter2:
                    r2 = len(inter2) / max(2.0, math.sqrt(max(1, len(eA) * len(aA2))))
                    n2 = len(eN & aN2) / max(1, len(eN))
                    l2 = len((eA - eN) & (aA2 - aN2)) / e_lats
                    base2 = 1.4 * r2 + 1.6 * n2 + 1.0 * l2 + 0.8 * len_score(enL[i - 1], arL[j - 2])
                    sc12 = base2 + (0.5 * sc11 if sc11 is not None else 0.0)
                    v = prev[j - 2] + sc12
                    if v > best: best, code = v, 2
            if i >= 2 and prevprev is not None and sc11 is not None:  # 2:1
                eA2, eN2 = enA[i - 2], enN[i - 2]
                inter_p = eA2 & aA
                if inter_p:
                    e_lats2 = max(1, len(eA2) - len(eN2))
                    rp = len(inter_p) / max(2.0, math.sqrt(max(1, len(eA2) * len(aA))))
                    np_ = len(eN2 & aN) / max(1, len(eN2))
                    lp = len((eA2 - eN2) & (aA - aN)) / e_lats2
                    sc21 = (1.4 * rp + 1.6 * np_ + 1.0 * lp + 0.8 * len_score(enL[i - 2], arL[j - 1])) + 0.5 * sc11
                    v = prevprev[j - 1] + sc21
                    if v > best: best, code = v, 5
            cur[j] = best
            bt[i, j] = code
        for j in range(hi + 1, m + 1):               # خارج النطاق (يمين): فجوات
            cur[j] = cur[j - 1] + GAP
            bt[i, j] = 4
        prevprev = prev[:]
        prev = cur

    i, j = n, m
    out = []
    while i > 0 or j > 0:
        if i == 0:
            out.append(((), (j - 1,))); j -= 1; continue
        if j == 0:
            out.append(((i - 1,), ())); i -= 1; continue
        code = bt[i, j]
        if code == 1:   out.append(((i - 1,), (j - 1,))); i -= 1; j -= 1
        elif code == 2: out.append(((i - 1,), (j - 2, j - 1))); i -= 1; j -= 2
        elif code == 5: out.append(((i - 2, i - 1), (j - 1,))); i -= 2; j -= 1
        elif code == 3: out.append(((i - 1,), ())); i -= 1
        else:           out.append(((), (j - 1,))); j -= 1
    out.reverse()

    res = []
    for ii, jj in out:
        if ii and jj:
            scs = []
            for a in ii:
                for b in jj:
                    eA, aA = enA[a], arA[b]
                    eN, aN = enN[a], arN[b]
                    inter = eA & aA
                    if not inter:
                        scs.append(-1.5); continue
                    e_lats = max(1, len(eA) - len(eN))
                    ratio = len(inter) / max(2.0, math.sqrt(max(1, len(eA) * len(aA))))
                    num_hit = len(eN & aN) / max(1, len(eN))
                    lat_hit = len((eA - eN) & (aA - aN)) / e_lats
                    scs.append(1.4 * ratio + 1.6 * num_hit + 1.0 * lat_hit + 0.8 * len_score(enL[a], arL[b]))
            res.append((ii, jj, float(np.mean(scs))))
        else:
            res.append((ii, jj, None))
    return res

def to_conf(sc):
    return round(max(0.0, min(1.0, sc / 2.2)), 3)

# ------------------------------------------------------------- المشغّل
def main():
    sents = json.load(open("sents.json", encoding="utf8"))
    t0 = time.time()
    print("== إقران الفصول ==", flush=True)
    pairs, unmatched = chapter_pairs(sents)
    rows = []
    for pi, (ch, lo, hi, hits) in enumerate(sorted(pairs, key=lambda p: p[1])):
        ix = [i for i, r in enumerate(sents["en"]) if r["chapter"] == ch]
        en_block = [sents["en"][i]["text"] for i in ix]
        ar_block = [r["text"] for r in sents["ar"][lo:hi]]
        print(f"  [{pi+1}/{len(pairs)}] {ch}: en={len(ix)} ar={hi-lo} ... ", flush=True)
        links = dp_align(en_block, ar_block)
        kept = 0
        for ii, jj, sc in links:
            if ii and jj and sc >= 0.55:
                kept += 1
                rows.append({
                    "chapter": ch.replace("_2026", ""),
                    "en": " ".join(en_block[a] for a in ii),
                    "ar": " ".join(ar_block[b] for b in jj),
                    "link_type": f"{len(ii)}:{len(jj)}",
                    "confidence": to_conf(sc),
                })
        print(f"     kept={kept} ({time.time()-t0:.0f}s)", flush=True)
    with open("aligned_pairs.csv", "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chapter", "en", "ar", "link_type", "confidence"])
        w.writeheader(); w.writerows(rows)
    print(f"WROTE aligned_pairs.csv rows={len(rows)} ({time.time()-t0:.0f}s)", flush=True)

def sample(k=50):
    rows = list(csv.DictReader(open("aligned_pairs.csv", encoding="utf8")))
    random.seed(7)
    picks = random.sample(rows, min(k, len(rows)))
    for i, r in enumerate(picks, 1):
        print(f"--- {i} [{r['chapter']}|{r['link_type']}|conf={r['confidence']}]")
        print("EN:", r["en"][:220])
        print("AR:", r["ar"][:220])

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        sample(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        main()
