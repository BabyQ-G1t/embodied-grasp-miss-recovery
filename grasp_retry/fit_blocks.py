#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 1108_All 上验收 blocks.py：不接相机、不动机械臂。

三个指标，全部有天然真值，不需要人工标注：

  ① 首帧召回   每集开始时两块都在蓝区 → 蓝区应检出 红1 灰1
  ② 末帧命中   每集结束灰方块应在黄区 → 黄区应检出 灰1
  ③ 空盘误检   每集开始时黄区是空的     → 黄区应检出 红0 灰0

③ 是这套改动的核心：旧判定在同一批首帧上给出 黄区灰 1296-5248 px，
   换成实例计数后必须归零，否则"没搬成功"会被判成 SUCCESS。

    python fit_blocks.py
    python fit_blocks.py --dataset OPQ_1108_ALL_36
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blocks  # noqa: E402

ROOT = pathlib.Path.home() / ".cache/huggingface/lerobot/shuren2"
BIN = (210, 116, 254, 161)
PICK = (208, 274, 262, 188)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="OPQ_1108_ALL_36")
    args = ap.parse_args()

    d = ROOT / args.dataset
    if not d.exists():
        print(f"[!] 找不到数据集 {d}")
        return 2
    eps = pd.concat([pd.read_parquet(f) for f in
                     glob.glob(str(d / "meta/episodes/**/*.parquet"), recursive=True)])
    vid = glob.glob(str(d / "videos/observation.images.F1/**/*.mp4"), recursive=True)[0]
    cap = cv2.VideoCapture(vid)

    def grab(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        return f if ok else None

    print(f"数据集 {args.dataset}   集数 {len(eps)}")
    print(f"黄区 {BIN}   蓝区 {PICK}\n")

    first_hit = 0
    last_hit = 0
    fp_red = fp_gray = 0
    n = 0
    for _, r in eps.iterrows():
        f0 = grab(r["videos/observation.images.F1/from_timestamp"] + 0.03)
        f1 = grab(r["videos/observation.images.F1/to_timestamp"] - 0.05)
        if f0 is None or f1 is None:
            continue
        n += 1

        pr, pg = blocks.count(f0, "red", PICK), blocks.count(f0, "gray", PICK)
        br, bg = blocks.count(f0, "red", BIN), blocks.count(f0, "gray", BIN)   # 空盘
        lr, lg = blocks.count(f1, "red", BIN), blocks.count(f1, "gray", BIN)   # 末帧

        if pr == 1 and pg == 1:
            first_hit += 1
        if lg >= 1:
            last_hit += 1
        if br or bg:
            fp_red += bool(br)
            fp_gray += bool(bg)

        ep = int(r["episode_index"])
        mark = "" if (pr == 1 and pg == 1 and not br and not bg) else "  <-- 异常"
        print(f"  ep{ep:02d} 首帧 蓝区 红{pr}灰{pg} | 空盘 红{br}灰{bg}   末帧 黄区 红{lr}灰{lg}{mark}")

    cap.release()
    print()
    print(f"① 首帧召回（蓝区 红1 灰1）      {first_hit}/{n}  = {first_hit / n * 100:.1f}%")
    print(f"② 末帧命中（黄区 灰≥1）        {last_hit}/{n}  = {last_hit / n * 100:.1f}%")
    print(f"③ 空盘误检 红                  {fp_red}/{n}   = {fp_red / n * 100:.1f}%")
    print(f"③ 空盘误检 灰                  {fp_gray}/{n}   = {fp_gray / n * 100:.1f}%")
    ok = first_hit / n >= 0.95 and last_hit / n >= 0.90 and fp_red / n <= 0.05 and fp_gray / n <= 0.05
    print("\n结论：" + ("达到门禁（召回≥95%、命中≥90%、误检≤5%）" if ok else "未达门禁，需调阈值"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
