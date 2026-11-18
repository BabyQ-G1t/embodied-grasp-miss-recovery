#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用版数据集合并：剔除指定集 -> 物理合并 N 个数据集 -> C1-C6 校验。

与 merge_datasets.py（OPQ_ALL_109 专用）的区别：源、剔除清单、产物名全部走命令行，
任意数据集个数，每个源可各有自己的剔除集。

用法（必须用 LeLab 的 venv）：
    PY=<你的 Python 解释器>
    $PY merge_1108.py --dry-run
    $PY merge_1108.py
    $PY merge_1108.py --verify-only

默认源（可用 --src 覆盖，格式 name:drop1,drop2 或 name:）：
    opq1108_20261108_124343:7   opq1108-1_20261108_124818:
产物：~/.cache/huggingface/lerobot/shuren2/OPQ_1108_ALL_36
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))
LOG_PATH = Path(__file__).with_name("merge_1108_log.txt")
PROV_DIR = Path(__file__).with_name("provenance_1108")
STATS_KEYS = ("observation.state", "action")

# 2026-11-08 复核结论（drop_scan.py 双判据 + 画面确认）：
#   1108 ep7 第一次闭合抓取未带起方块（段内位移 18px，其余 36 集 100~191px），
#   夹爪张开重抓，是全批唯一 3 段开合的集 -> 剔除。
#   1108-1 全部 29 集正常（末集 ep28：2 段开合、段内位移 126px、末帧黄板 57px）。
SOURCES: list[tuple[str, list[int]]] = [
    ("opq1108_20261108_124343", [7]),
    ("opq1108-1_20261108_124818", []),
]
AGG_NAME = "OPQ_1108_ALL_36"

_log: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    _log.append(msg)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_log) + "\n", encoding="utf-8")


def src_root(name: str) -> Path:
    return LEROBOT_HOME / "shuren2" / name


def clean_name(name: str) -> str:
    return f"{name}_clean"


def agg_root() -> Path:
    return LEROBOT_HOME / "shuren2" / AGG_NAME


# --------------------------------------------------------------------------- #
# 元数据
# --------------------------------------------------------------------------- #
def episode_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"找不到 episode 元数据: {root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def lens_of(root: Path) -> list[int] | None:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index")["length"].astype(int).tolist()


def dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def fingerprint(root: Path) -> dict:
    info_p = root / "meta" / "info.json"
    if not info_p.exists():
        return {"exists": False}
    info = json.loads(info_p.read_text(encoding="utf-8"))
    lens = lens_of(root)
    files = []
    for sub in ("data", "videos"):
        d = root / sub
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    files.append(f"{p.relative_to(root).as_posix()}:{p.stat().st_size}")
    h = hashlib.sha1(json.dumps([info.get("total_episodes"), info.get("total_frames"),
                                 info.get("fps"), sorted(info.get("features", {})),
                                 lens, files]).encode()).hexdigest()[:16]
    return {"exists": True, "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"), "len_seq": lens,
            "n_files": len(files), "sha1": h}


def expected_lens(kept_sources: list[tuple[str, list[int]]]) -> list[int]:
    """合并后逐集帧数：按源顺序拼接，跳过各自的剔除集。clean 目录已存在则读 clean。"""
    seq: list[int] = []
    for name, drop in kept_sources:
        root = src_root(clean_name(name)) if drop else src_root(name)
        if drop and not (root / "meta" / "info.json").exists():
            root = src_root(name)
        df = episode_table(root)
        for _, r in df.iterrows():
            if int(r["episode_index"]) in drop and root == src_root(name):
                continue
            seq.append(int(r["length"]))
    return seq


def agg_inputs(kept_sources: list[tuple[str, list[int]]]) -> list[tuple[str, Path, int]]:
    out = []
    for name, drop in kept_sources:
        root = src_root(clean_name(name)) if drop else src_root(name)
        lens = lens_of(root) or []
        out.append((name, root / "meta" / "stats.json", sum(lens)))
    return out


def expected_stats(items: list[tuple[str, Path, int]]) -> dict:
    res = {}
    for key in STATS_KEYS:
        mins, maxs, means, stds, ns = [], [], [], [], []
        for _n, p, n in items:
            s = json.loads(p.read_text(encoding="utf-8"))[key]
            mins.append(np.array(s["min"], float))
            maxs.append(np.array(s["max"], float))
            means.append(np.array(s["mean"], float))
            stds.append(np.array(s["std"], float))
            ns.append(float(n))
        w = np.array(ns)[:, None]
        total = w.sum()
        mean = (np.array(means) * w).sum(0) / total
        var = (np.array([sd ** 2 + mu ** 2 for sd, mu in zip(stds, means)]) * w).sum(0) / total - mean ** 2
        res[key] = {"min": np.min(mins, 0), "max": np.max(maxs, 0),
                    "mean": mean, "std": np.sqrt(np.maximum(var, 0.0))}
    return res


def src_fingerprints(kept_sources: list[tuple[str, list[int]]]) -> dict:
    out = {}
    for name, drop in kept_sources:
        fp = fingerprint(src_root(name))
        out[name] = {"sha1": fp.get("sha1"), "episodes": fp.get("episodes"),
                     "frames": fp.get("frames"), "drop": drop}
    return out


def save_prov(kind: str, root: Path, exp: list[int], srcs: dict) -> None:
    PROV_DIR.mkdir(exist_ok=True)
    fp = fingerprint(root)
    (PROV_DIR / f"{kind}.json").write_text(json.dumps({
        "kind": kind, "root": str(root), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "merge_1108.py", "src_fps": srcs,
        "expected": {"episodes": len(exp), "frames": sum(exp)},
        "out": {"episodes": fp.get("episodes"), "frames": fp.get("frames"), "sha1": fp.get("sha1")},
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def reusable(root: Path, kind: str, exp: list[int], srcs: dict, rebuild: bool) -> tuple[bool, str]:
    if rebuild or not (root / "meta" / "info.json").exists():
        return False, "强制重建" if rebuild else "目标不存在"
    if not any((root / "videos").rglob("*.mp4")) or not any((root / "data").rglob("*.parquet")):
        return False, "结构不完整（缺视频或数据分片）"
    got = lens_of(root)
    if not got or len(got) != len(exp):
        return False, f"集数 {len(got or [])}/{len(exp)}"
    diff = [i for i, (a, b) in enumerate(zip(got, exp)) if a != b]
    if diff:
        return False, f"逐集帧数不一致（首个第 {diff[0]} 集 {got[diff[0]]}/{exp[diff[0]]}）"
    p = PROV_DIR / f"{kind}.json"
    if not p.exists():
        return False, "无来源凭据"
    prov = json.loads(p.read_text(encoding="utf-8"))
    if prov.get("src_fps") != srcs:
        return False, "源指纹变化"
    if prov.get("out", {}).get("sha1") != fingerprint(root)["sha1"]:
        return False, "产物指纹变化"
    return True, "凭据 + 逐集长度 + 文件清单一致"


# --------------------------------------------------------------------------- #
# 阶段 A：剔除
# --------------------------------------------------------------------------- #
def build_clean(name: str, drop: list[int], srcs: dict, dry: bool, rebuild: bool) -> Path:
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = src_root(clean_name(name))
    exp = [int(r["length"]) for _, r in episode_table(src_root(name)).iterrows()
           if int(r["episode_index"]) not in drop]
    if root.exists():
        ok, why = reusable(root, clean_name(name), exp, srcs, rebuild)
        if ok:
            log(f"[A] 复用 {clean_name(name)}（{len(exp)} 集 / {sum(exp)} 帧）：{why}")
            return root
        log(f"[A] 删除重建 {clean_name(name)}：{why}")
        if dry:
            return root
        shutil.rmtree(root)
    if dry:
        log(f"[A] 将剔除 {name} 的 episode {drop} -> {root}")
        return root

    log(f"[A] 剔除 {name} 的 episode {drop} ...")
    t0 = time.time()
    ds = LeRobotDataset(repo_id=f"shuren2/{name}", root=src_root(name))
    delete_episodes(ds, episode_indices=drop, output_dir=root, repo_id=f"shuren2/{clean_name(name)}")
    log(f"[A] 完成 {time.time() - t0:.1f}s")
    got = lens_of(root)
    if got != exp:
        raise RuntimeError(f"[A] 逐集帧数不符：{got} vs {exp}")
    save_prov(clean_name(name), root, exp, srcs)
    return root


# --------------------------------------------------------------------------- #
# 阶段 B：合并
# --------------------------------------------------------------------------- #
def aggregate(kept: list[tuple[str, list[int]]], srcs: dict, dry: bool, rebuild: bool) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets

    root = agg_root()
    repo_ids = [f"shuren2/{clean_name(n)}" if d else f"shuren2/{n}" for n, d in kept]
    roots = [src_root(clean_name(n)) if d else src_root(n) for n, d in kept]
    exp = expected_lens(kept)

    if root.exists():
        ok, why = reusable(root, AGG_NAME, exp, srcs, rebuild)
        if ok:
            log(f"[B] 复用 {AGG_NAME}（{len(exp)} 集 / {sum(exp)} 帧）：{why}")
            return
        log(f"[B] 删除重建 {AGG_NAME}：{why}")
        if dry:
            return
        shutil.rmtree(root)
    log("[B] 合并源：")
    for r, p in zip(repo_ids, roots):
        log(f"      {r}  <-  {p}")
    if dry:
        log(f"[B] 将输出到 {root}")
        return

    root.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    aggregate_datasets(repo_ids=repo_ids, aggr_repo_id=f"shuren2/{AGG_NAME}",
                       roots=[Path(p) for p in roots], aggr_root=root,
                       data_files_size_in_mb=100, video_files_size_in_mb=200,
                       concatenate_videos=True, concatenate_data=True)
    log(f"[B] 完成 {time.time() - t0:.1f}s")
    got = lens_of(root)
    if got != exp:
        raise RuntimeError(f"[B] 逐集帧数不符：{len(got or [])} 集 vs 期望 {len(exp)} 集")
    save_prov(AGG_NAME, root, exp, srcs)
    log(f"[B] 已写凭据 provenance_1108/{AGG_NAME}.json")


# --------------------------------------------------------------------------- #
# 阶段 C：校验
# --------------------------------------------------------------------------- #
def verify(kept: list[tuple[str, list[int]]], skip_frames: bool = False) -> bool:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch

    root = agg_root()
    ok = True

    def check(cond: bool, title: str, detail: str = "") -> None:
        nonlocal ok
        log(f"  {'✔' if cond else '✘'} {title}" + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    log("[C] 校验开始")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    exp = expected_lens(kept)

    check(info["total_episodes"] == len(exp), "C1 集数", f"实际 {info['total_episodes']} / 期望 {len(exp)}")
    check(info["total_frames"] == sum(exp), "C1 帧数", f"实际 {info['total_frames']} / 期望 {sum(exp)}")
    check(info["fps"] == 30 and info["robot_type"] == "so_follower",
          "C1 fps 与机型", f"fps={info['fps']} robot={info['robot_type']}")
    feats = info["features"]
    check(all(k in feats and feats[k]["dtype"] == "video"
              for k in ("observation.images.F1", "observation.images.F2")),
          "C1 双相机字段齐全", f"{[k for k in feats if k.startswith('observation.images')]}")
    check(tuple(feats["action"]["shape"]) == (6,) and tuple(feats["observation.state"]["shape"]) == (6,),
          "C1 action/state 维度均为 6")

    ep = episode_table(root)
    got = ep["length"].astype(int).tolist()
    if len(got) != len(exp):
        check(False, "C2 逐集帧数与源一致", f"集数 {len(got)}/{len(exp)}")
    else:
        diff = [i for i, (a, b) in enumerate(zip(got, exp)) if a != b]
        check(not diff, "C2 逐集帧数与源一致",
              "" if not diff else f"首个不一致第 {diff[0]} 集 {got[diff[0]]}/{exp[diff[0]]}")

    fps = info["fps"]
    f1f = ep["videos/observation.images.F1/from_timestamp"].to_numpy()
    f1t = ep["videos/observation.images.F1/to_timestamp"].to_numpy()
    f2f = ep["videos/observation.images.F2/from_timestamp"].to_numpy()
    f2t = ep["videos/observation.images.F2/to_timestamp"].to_numpy()
    d1, d2 = f1t - f1f, f2t - f2f
    check(np.abs(d1 - d2).max() < 1e-3, "C3 F1/F2 逐集时长一致（仅元数据层面）",
          f"最大差 {np.abs(d1 - d2).max():.2e}s")

    for key, fr, to, dl in (("F1", f1f, f1t, d1), ("F2", f2f, f2t, d2)):
        ch = ep[f"videos/observation.images.{key}/chunk_index"].to_numpy()
        fi = ep[f"videos/observation.images.{key}/file_index"].to_numpy()
        bad = [i for i in range(1, len(ep))
               if ch[i] == ch[i - 1] and fi[i] == fi[i - 1] and abs(fr[i] - to[i - 1]) > 1e-3]
        check(not bad, f"C4 {key} 同分片内首尾相接", "" if not bad else f"首个断点第 {bad[0]} 集")
        check((fr >= -1e-6).all(), f"C4 {key} 跨分片时间戳非负")
        check(np.abs(dl - ep["length"].to_numpy() / fps).max() < 0.05,
              f"C4 {key} 时长与帧数自洽",
              f"最大误差 {np.abs(dl - ep['length'].to_numpy() / fps).max() * 1000:.1f}ms")
        per = pd.Series([f"{c}/{f}" for c, f in zip(ch, fi)]).value_counts().sort_index().to_dict()
        log(f"      {key} 分片(chunk/file): {per}")

    if not skip_frames:
        ds = LeRobotDataset(f"shuren2/{AGG_NAME}", root=root)
        starts = ep["dataset_from_index"].astype(int).tolist()
        picks = sorted({starts[0], starts[len(starts) // 2], starts[-1]})
        picks = [p for p in picks if 0 <= p < ds.meta.total_frames]
        for idx in picks:
            s = ds[idx]
            imgs = [k for k in s if k.startswith("observation.images.")]
            good = (len(imgs) == 2
                    and all(tuple(s[k].shape) == (3, 480, 640) for k in imgs)
                    and all(torch.isfinite(s[k].float()).all().item() for k in imgs)
                    and tuple(s["observation.state"].shape) == (6,)
                    and tuple(s["action"].shape) == (6,))
            check(good, f"C5 抽样读取 index={idx}",
                  f"画面 {len(imgs)} 路，首路 {tuple(s[imgs[0]].shape) if imgs else '-'}")
        del ds

    st = json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))
    items = agg_inputs(kept)
    missing = [str(p) for _, p, _ in items if not p.exists()]
    if missing:
        check(False, "C6 合并输入的 stats 齐全", f"缺 {missing}")
    else:
        exp_s = expected_stats(items)
        for key in STATS_KEYS:
            for stat in ("min", "max", "mean", "std"):
                a = np.array(st.get(key, {}).get(stat, []), float)
                e = exp_s[key][stat]
                good = a.shape == e.shape and np.allclose(a, e, rtol=1e-4, atol=1e-3)
                check(good, f"C6 {key} {stat}",
                      f"最大差 {np.abs(a - e).max():.2e}" if a.shape == e.shape else f"维度 {a.shape}")
    log(f"[C] 校验{'通过' if ok else '未通过'}")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    global SOURCES, AGG_NAME
    ap = argparse.ArgumentParser(description="通用数据集剔除 + 合并 + 校验")
    ap.add_argument("--src", action="append", default=None,
                    help="源数据集目录名，可带剔除集，如 opq1108_20261108_124343:7")
    ap.add_argument("--agg", default=None, help=f"产物名，默认 {AGG_NAME}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-frames", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.src:
        SOURCES = [(s.split(":")[0], [int(x) for x in s.split(":")[1].split(",") if x.strip()])
                   if ":" in s else (s, []) for s in args.src]
    if args.agg:
        AGG_NAME = args.agg

    log("=" * 66)
    log(f"{AGG_NAME} 合并  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"缓存根目录 {LEROBOT_HOME}")
    for name, drop in SOURCES:
        log(f"      源 {name}  剔除 {drop or '无'}")

    try:
        if args.verify_only:
            return 0 if verify(SOURCES, args.skip_frames) else 1

        srcs = src_fingerprints(SOURCES)
        for name, v in srcs.items():
            log(f"      源指纹 {name}: {v['sha1']}  {v['episodes']} 集 / {v['frames']} 帧")

        for name, drop in SOURCES:
            if drop:
                build_clean(name, drop, srcs, args.dry_run, args.rebuild)
        aggregate(SOURCES, srcs, args.dry_run, args.rebuild)

        if args.dry_run:
            exp = expected_lens(SOURCES)
            tot = sum(dir_size_mb(src_root(n)) for n, _ in SOURCES)
            log(f"[dry-run] 剔除后 {len(exp)} 集 / {sum(exp)} 帧，源 {tot:.0f} MB，未写任何文件")
            return 0

        ok = verify(SOURCES, args.skip_frames)
        log(f"产物：{agg_root()}  ({dir_size_mb(agg_root()):.0f} MB)")
        return 0 if ok else 1
    except Exception:
        log("[失败] " + traceback.format_exc())
        return 2
    finally:
        if not args.dry_run:
            flush_log()


if __name__ == "__main__":
    sys.exit(main())
