#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anchor-based DP sentence alignment v2 (stricter matching).

Key fix vs v1: shared tiny list indices (1,2,3) no longer count as evidence.
Distinctive anchors = Latin terms (incl. C7/T1/L5...) + numbers that are
long/decimal/percent (0.9, 30, 50%, 235). A link with NO shared distinctive
anchor is heavily penalized, so pure list-number coincidences get gapped.
"""
import re, math
import numpy as np

LATIN = re.compile(r"[A-Za-z][A-Za-z0-9\-']{1,}")
NUM = re.compile(r"\d+(?:[.,]\d+)*\s*%?")
STOP_LATIN = set("""table figure figures chapter updated summary created https www
page pages et al fig figs no nos vs see above below following use used using
key point points question questions answer answers choice choices concept tested""".split())

TINY = {"1", "2", "3", "4"}

def anchors(s):
    lat = {t.lower() for t in LATIN.findall(s)} - STOP_LATIN
    return lat | set(NUM.findall(s))

def _nums(A):
    return {a for a in A if a[0].isdigit()}

def _distinct(A):
    """Distinctive numbers: 2+ digits, decimals, or 5-9/0 single digits."""
    return {a for a in _nums(A) if len(a) > 1 or a in {"0", "5", "6", "7", "8", "9"}}

GAP = -0.55
MU, SIG = 1.15, 0.45

def len_score(le, la):
    r = math.log((la / le) / MU)
    return max(-1.5, -0.5 * (r / SIG) ** 2)

def pscore(eA, aA, eN, aN, le, la):
    """Pair score: Latin overlap + distinctive-number overlap + length prior."""
    eL = eA - eN; aL = aA - aN
    eD = _distinct(eN); aD = _distinct(aN)
    lat_inter = eL & aL
    num_inter = eD & aD
    if not lat_inter and not num_inter:
        return -1.5
    ratio = len(lat_inter | num_inter) / max(2.0, math.sqrt(max(1, len(eA) * len(aA))))
    num_hit = len(num_inter) / max(1, len(eD))
    penalty = -1.2 if (eD and not num_inter) else 0.0
    lat_hit = len(lat_inter) / max(1, len(eL))
    return 1.4 * ratio + 1.6 * num_hit + 1.0 * lat_hit + 0.8 * len_score(le, la) + penalty

def dp_align(en_s, ar_s, band_frac=0.06, min_band=250):
    n, m = len(en_s), len(ar_s)
    if n == 0 or m == 0:
        return []
    enA = [anchors(s) for s in en_s]; arA = [anchors(s) for s in ar_s]
    enN = [_nums(x) for x in enA]; arN = [_nums(x) for x in arA]
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
        for j in range(1, lo):
            cur[j] = cur[j - 1] + GAP
            bt[i, j] = 4
        eA, eN = enA[i - 1], enN[i - 1]
        for j in range(lo, hi + 1):
            best, code = NEG, 3
            v = prev[j] + GAP                        # 1:0
            if v > best: best, code = v, 3
            aA, aN = arA[j - 1], arN[j - 1]
            sc11 = pscore(eA, aA, eN, aN, enL[i - 1], arL[j - 1])
            v = prev[j - 1] + sc11                   # 1:1
            if v > best: best, code = v, 1
            v = cur[j - 1] + GAP                     # 0:1
            if v > best: best, code = v, 4
            if j >= 2:                               # 1:2
                sc12 = 0.5 * (pscore(eA, arA[j - 2], eN, arN[j - 2], enL[i - 1], arL[j - 2]) + sc11)
                v = prev[j - 2] + sc12
                if v > best: best, code = v, 2
            if i >= 2 and prevprev is not None:      # 2:1
                sc21 = 0.5 * (pscore(enA[i - 2], aA, enN[i - 2], aN, enL[i - 2], arL[j - 1]) + sc11)
                v = prevprev[j - 1] + sc21
                if v > best: best, code = v, 5
            cur[j] = best
            bt[i, j] = code
        for j in range(hi + 1, m + 1):
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
            scs = [pscore(enA[a], arA[b], enN[a], arN[b], enL[a], arL[b]) for a in ii for b in jj]
            res.append((ii, jj, float(np.mean(scs))))
        else:
            res.append((ii, jj, None))
    return res

def to_conf(sc):
    return round(max(0.0, min(1.0, sc / 2.2)), 3)
