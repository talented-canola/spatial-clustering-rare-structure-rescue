"""一键把 code/ 下所有脚本里的硬编码路径改写到 config.py 指定的新位置。

用法：
    1. 编辑 config.py，改好 PROJECT_ROOT / WORK_DIR / DATA_H5AD / RESULTS_DIR
    2. python port_paths.py --check     # 先看会改哪些，不落盘
    3. python port_paths.py             # 确认无误后执行

改造是纯文本前缀替换，可逆：原前缀在脚本中被替换为新前缀，
"--revert" 用旧路径文件（.port_paths_backup）回滚。
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# 原前缀 → config 中的目标路径
# 注意顺序：长的前缀必须先替换，否则短前缀会先把长前缀切碎
RULES = [
    ("F:/BGI/task3/spateo-release-main/data", config.DATA_H5AD.rsplit("/", 1)[0]),
    ("F:/BGI/task3/spateo-release-main/results", config.RESULTS_DIR.replace("\\", "/")),
    ("F:/BGI/task3/spateo-release-main", config.PROJECT_ROOT.replace("\\", "/")),
    ("F:/BGI/task3/baseline_results", config.WORK_DIR.replace("\\", "/") + "/baseline_results"),
    ("F:/tmp", config.WORK_DIR.replace("\\", "/")),
]

BACKUP = ".port_paths_backup"
CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")


def iter_py():
    for d, _, fs in os.walk(CODE_DIR):
        for f in fs:
            if f.endswith(".py"):
                yield os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只预览，不写入")
    ap.add_argument("--revert", action="store_true", help="从备份回滚")
    args = ap.parse_args()

    if args.revert:
        if not os.path.isdir(BACKUP):
            sys.exit(f"找不到备份目录 {BACKUP}，无法回滚。")
        for d, _, fs in os.walk(BACKUP):
            for f in fs:
                src = os.path.join(d, f)
                dst = os.path.join(CODE_DIR, os.path.relpath(src, BACKUP))
                shutil.copy2(src, dst)
        print(f"已从 {BACKUP} 回滚全部脚本。")
        return

    if not args.check and not os.path.isdir(BACKUP):
        shutil.copytree(CODE_DIR, BACKUP)
        print(f"已备份 code/ → {BACKUP}/")

    total_files, total_hits = 0, 0
    for p in iter_py():
        s = orig = open(p, encoding="utf-8").read()
        hits = 0
        for old, new in RULES:
            if old == new:
                continue
            n = s.count(old)
            if n:
                s = s.replace(old, new)
                hits += n
        if hits:
            total_files += 1
            total_hits += hits
            rel = os.path.relpath(p, os.path.dirname(CODE_DIR))
            print(f"  {'[预览] ' if args.check else ''}{rel}  ({hits} 处)")
            if not args.check:
                open(p, "w", encoding="utf-8").write(s)

    print(f"\n{'将修改' if args.check else '已修改'} {total_files} 个文件，共 {total_hits} 处路径。")
    if args.check:
        print("确认无误后运行： python port_paths.py")


if __name__ == "__main__":
    main()
