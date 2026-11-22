#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OrthoAlign CLI — محاذاة جمل إنجليزي/عربي بين كتب PDF.

أمثلة:
  python cli_app.py extract  ./books/en ./books/ar ./work2      # استخراج الجمل
  python cli_app.py align    ./work2 ./output                   # المحاذاة الكاملة + CSV
  python cli_app.py search   ./output/labse_retrieval.csv "ACL" # بحث في النتائج

المتطلبات: poppler-utils (أمر pdftotext) للمرحلة الأولى، وtorch +
sentence-transformers للمحاذاة الكاملة (GPU موصى به لـ81 ألف جملة).
"""
import csv
import glob
import shutil
import subprocess
import sys
from pathlib import Path


def cmd_extract(en_dir, ar_dir, out_dir):
    """pdftotext لكل ملفات PDF ثم بناء sents.json في out_dir."""
    if shutil.which("pdftotext") is None:
        sys.exit("pdftotext غير موجود — ثبّت poppler-utils: sudo apt install poppler-utils")
    out = Path(out_dir)
    (out / "en").mkdir(parents=True, exist_ok=True)
    (out / "ar").mkdir(parents=True, exist_ok=True)
    for f in glob.glob(f"{en_dir}/*.pdf"):
        b = Path(f).stem.replace(" ", "_")
        subprocess.run(["pdftotext", "-layout", f, str(out / "en" / f"{b}.txt")], check=False)
    for f in glob.glob(f"{ar_dir}/*.pdf"):
        b = Path(f).stem.replace(" ", "_")
        subprocess.run(["pdftotext", "-layout", f, str(out / "ar" / f"{b}.txt")], check=False)
    import prepare
    prepare.main(str(out), str(out / "sents.json"))
    print("OK:", str((out / "sents.json").resolve()))


def cmd_align(work_dir, out_dir):
    """المحاذاة الكاملة: التضمينات (إن غابت) + الاسترجاع + إقران المناطق + DP."""
    import run_align
    run_align.main(work_dir, out_dir)


def cmd_search(csv_path, query, limit=20):
    q = query.lower()
    n = 0
    with open(csv_path, encoding="utf8") as f:
        for row in csv.DictReader(f):
            if q in row["en"].lower() or q in row.get("ar", "").lower():
                print(f"[{row['chapter']}|{row.get('confidence', '')}] {row['en'][:110]}")
                print(f"   <=> {row.get('ar', '')[:110]}\n")
                n += 1
                if n >= limit:
                    break
    if not n:
        print("لا نتائج")


if __name__ == "__main__":
    cmds = {"extract": cmd_extract, "align": cmd_align, "search": cmd_search}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    cmds[sys.argv[1]](*sys.argv[2:])
