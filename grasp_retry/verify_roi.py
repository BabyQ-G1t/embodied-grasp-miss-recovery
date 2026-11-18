#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线验证判定层：不接相机、不动机械臂。

做法：拿一张静态环境照当"执行前帧"，用图像编辑伪造三种"执行后帧"，
      再跑 run_trial.py 里真实的 mask / find_blocks / unit_px_of /
      check_boxes / judge，看判定是否符合预期。

    python verify_roi.py                      # 红块，默认用 ../1108最新环境照.png
    python verify_roi.py --color gray
    python verify_roi.py --image path/to/frame.png

三种场景：
    success  块从蓝区消失 + 黄区出现  -> SUCCESS
    lost     块从蓝区消失，黄区没出现 -> LOST
    none     什么都没动               -> EMPTY

为什么要有这个脚本：
    judge 依赖 bin 框和颜色阈值，改标定后必须能离线复验，
    否则只能等真机跑一轮才知道对错，而那时候错的代价是一整轮试验作废。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blocks  # noqa: E402
import run_trial as rt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

# 两块积木在 1108 环境照里的位置（用于伪造执行后帧）
PATCH = {
    "red":  (230, 308, 44, 44),
    "gray": (398, 304, 42, 45),
}


def mask(img, color):
    # 阈值统一在 blocks.py，避免两处各写一份、改一处漏一处
    return blocks.mask(img, color)


def fill_blue(img, box):
    """把 box 区域涂成蓝板底色，模拟"块被拿走"。"""
    x, y, w, h = box
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    ring = img[max(0, y - 12):y, x:x + w].reshape(-1, 3)
    col = np.median(ring, axis=0).astype(np.uint8) if ring.size else np.array([190, 120, 60], np.uint8)
    img[y0:y1, x0:x1] = col
    return img


def paste(img, patch, center):
    ph, pw = patch.shape[:2]
    cx, cy = center
    x, y = cx - pw // 2, cy - ph // 2
    img[y:y + ph, x:x + pw] = patch
    return img


def run(img_pre, img_post, box_target, keep_ratio=0.40):
    roi = rt.load_bin_roi(None)
    m_pre, m_post = mask(img_pre, box_target["color"]), mask(img_post, box_target["color"])

    cands = rt.find_blocks(m_pre)
    table = [b for b in cands if rt.overlap_ratio(b, roi) < 0.5]
    boxes = rt.select_blocks(table, 1)
    unit_px = rt.unit_px_of(m_pre, boxes)
    bin_px0 = rt.px_in_box(m_pre, roi)
    bin_px1 = rt.px_in_box(m_post, roi)
    status = rt.check_boxes(m_pre, m_post, boxes, keep_ratio)
    verdict = rt.judge(status, bin_px1 - bin_px0, unit_px)
    return dict(boxes=boxes, unit_px=unit_px, bin0=bin_px0, bin1=bin_px1,
                gain=bin_px1 - bin_px0, status=status, verdict=verdict,
                need=rt.BIN_GAIN_UNIT * unit_px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(HERE.parent / "1108最新环境照.png"))
    ap.add_argument("--color", default="red", choices=("red", "gray"))
    args = ap.parse_args()

    path = pathlib.Path(args.image)
    if not path.exists():
        print(f"[!] 找不到图：{path}")
        return 2
    base = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    roi = rt.load_bin_roi(None)
    print(f"图片 {path.name}   颜色路 {args.color}")
    print(f"收纳区 bin={roi}  (254x161)")
    if roi is None:
        print("[!] bin_roi.json 读不到")
        return 2

    px, py, pw, ph = PATCH[args.color]
    patch = base[py:py + ph, px:px + pw].copy()

    # 场景 A：抓走且放进黄区中心
    a = base.copy()
    fill_blue(a, PATCH[args.color])
    paste(a, patch, (roi[0] + roi[2] // 2, roi[1] + roi[3] // 2))

    # 场景 B：抓走但没进黄区（掉桌外 / 推走）
    b = base.copy()
    fill_blue(b, PATCH[args.color])

    # 场景 C：什么都没动
    c = base.copy()

    print(f"单块面积 {pw * ph} px（红块实际色像素约 1467）\n")
    expect = {"A 抓走并放进": ("success", a, "SUCCESS"),
              "B 抓走但没进区": ("lost", b, "LOST"),
              "C 完全没动": ("none", c, "EMPTY")}
    ok = True
    for name, (_, img, want) in expect.items():
        r = run(base, img, dict(color=args.color))
        got = r["verdict"]
        flag = "PASS" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"[{flag}] {name:14s} 判定={got:8s} 期望={want:8s} "
              f"原位框={r['boxes']} 单块当量={r['unit_px']:.0f}px")
        print(f"        收纳区 {r['bin0']} -> {r['bin1']} px  增量 {r['gain']:+d}  "
              f"门限 {r['need']:.0f}  相对占比={[round(s['rel'], 2) for s in r['status']]}")
    print("\n结论：" + ("全部符合预期" if ok else "有不符预期的场景，判定层需修"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
