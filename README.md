# OrthoAlign — محاذاة الجمل بين كتاب Orthobullets الإنجليزي (2026) والعربي (2021)

مشروع محاذاة جمل كتاب Orthobullets الإنجليزي مع ترجمته العربية، وإنتاج جدول CSV
يضم كل جملة إنجليزية ومقابلها العربي مع درجة ثقة.

## المخرجات
| الملف | الوصف |
|---|---|
| `labse_retrieval.csv` | النتيجة المعتمدة: 2,200 زوج مع confidence وmargin |
| `aligned_pairs.csv` | نتيجة DP بالمراسي (تجريبي) |
| `spotcheck50.txt` / `qc_retrieval.txt` | نماذج الفحص العشوائي المقاس |

## الكود
- `prepare.py` — التقطيع والتطبيع (NFKC، حماية الاختصارات، الأرقام العربية-الهندية)
- `pair_regions.py` — إقران الفصول بالمحتوى (LaBSE + خوارزمية الهنغارية)
- `align.py` — الخوارزمية الكاملة (المراحل الثلاث موثقة داخل الملف)
- `dp0.py` — محاذاة DP بالمراسي (1:1 / 1:2 / 2:1 / 1:0 / 0:1)
- `final_align.py` — مشغّل المحاذاة النهائية
- `cli_app.py` — تطبيق أوامر: extract / align / search
- `README_AR.md` — التوثيق الكامل بالعربية

## التشغيل
```bash
pip install sentence-transformers scipy numpy
python3 cli_app.py extract ./books/en ./books/ar ./work2
python3 cli_app.py align ./work2 ./out
python3 cli_app.py search labse_retrieval.csv "ACL"
```

الدقة المقاسة بفحص عشوائي: صفوف cos ≥ 0.75 وmargin ≥ 0.05 ≈ 96% (عينة 25)؛
كل صف يحمل confidence — لا تُعتمد الصفوف الضعيفة دون مراجعة بشرية.
