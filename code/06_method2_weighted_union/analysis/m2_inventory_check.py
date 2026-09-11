"""Inventory check: validate experiment_registry.csv path columns against the filesystem."""
import os, pandas as pd

ROOT = r"F:/BGI/task3/spateo-release-main/results"
CSV = os.path.join(ROOT, "experiment_registry.csv")

df = pd.read_csv(CSV, keep_default_na=False, encoding="utf-8-sig")
df["resolution"] = df["resolution"].astype(str).str.strip()
print("registry rows:", len(df))
print()

path_cols = ["完整19类重叠表路径", "空间图路径", "混淆矩阵路径", "UMAP路径", "marker基因验证路径"]
rec_missing = 0        # cell literally says 缺失
not_found = []         # cell names a file that does not exist
ok_existing = []       # cell names a file that exists

print("=" * 110)
for r in df.itertuples(index=False):
    rid, res = str(r.方案ID), str(r.resolution)
    line_parts = []
    for col in path_cols:
        v = str(getattr(r, col)).strip()
        if v == "" or v.lower() == "nan":
            v = "缺失"
        if v == "缺失":
            rec_missing += 1
            line_parts.append(f"[{col[:4]}=登记缺失]")
        else:
            p = os.path.join(ROOT, v)
            if os.path.exists(p):
                ok_existing.append(v)
                line_parts.append(f"[{col[:4]}=存在]")
            else:
                not_found.append((rid, res, col, v))
                line_parts.append(f"[{col[:4]}=**不存在:{v}**]")
    print(f"{rid:<28} res={res:<5} " + "  ".join(line_parts))
print("=" * 110)
print()
print("cells marked 缺失 in registry:", rec_missing)
print("path cells that EXIST:", len(ok_existing))
print("path cells that POINT TO MISSING FILES:", len(not_found))
if not_found:
    print("  MISSING-FILE DETAILS (row: resolution, column, path):")
    for rid, res, col, v in not_found:
        print(f"    {rid:<28} res={res:<5} {col}: {v}")
print()

# folder-level completeness of the four-piece set: 空间图 + 混淆矩阵 + 完整重叠表(matching) + README
print("=" * 110)
print("FOLDER-LEVEL four-piece completeness (per CLAUDE.md convention)")
print("=" * 110)
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p) or d in ("analysis",):
        continue
    figs = os.path.join(p, "figures")
    tabs = os.path.join(p, "tables")
    readme = os.path.exists(os.path.join(p, "README.md"))
    has_spatial = any(f.startswith("spatial_aligned") or f.startswith("raw_spatial") for f in os.listdir(figs)) if os.path.isdir(figs) else False
    has_confusion = any(f.startswith("confusion") for f in os.listdir(figs)) if os.path.isdir(figs) else False
    has_matching = any(f.startswith("matching") for f in os.listdir(tabs)) if os.path.isdir(tabs) else False
    pieces = sum([has_spatial, has_confusion, has_matching, readme])
    print(f"{d:<30} spatial={int(has_spatial)} confusion={int(has_confusion)} matching={int(has_matching)} README={int(readme)}  -> {pieces}/4")
