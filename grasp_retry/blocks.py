#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""积木实例检测：把"框内像素总量"换成"框内有几个方块实例"。

为什么必须换：
    旧判定用 px_in_box 数像素。1108_All 实测：
      · 机械臂遮到黄板时，黄区红像素从 26 飙到 10182（机械臂橙色外壳）
      · 黄板被阴影压暗时，黄区灰像素从 200 涨到 5781（黄板自身命中灰掩码）
    两条都会把"没搬成功"判成 SUCCESS。改成实例计数后，同一批帧上
    噪声实例数全部归零，因为橙色外壳是长条、阴影是散斑，过不了方块形状。

两类积木的实测特征（1108_All 全部 36 集首帧，见 fit_blocks.py）：
    红圆柱 面积 1273-1308  宽 40-42  高 42    填充率 0.72-0.77（圆，理论 π/4）
    灰方块 面积  878-1856  宽 36-50  高 36-56 填充率 0.51-0.94（方，会被斜放）

验收（fit_blocks.py，1108_All 36 集）：
    首帧蓝区召回 100%   末帧黄区命中 100%   空盘误检 0%
    留出验证：偶数集选参 → 奇数集同样 100%/100%/0%，不是过拟合
"""
from __future__ import annotations

import cv2
import numpy as np

# 颜色掩码
#   红：V>70 不可省。纯黑像素 HSV=(0,255,低V)，不加 V 下限会把夹爪判成红色。
#   灰：这个方块是偏蓝的深灰（H≈113 S≈94 V≈120），与蓝板（H≈109 S≈162 V≈200）
#       同色相，只能靠 S 和 V 区分。S<70 会完全漏检。
THR = {
    "red":  dict(h_lo=170, h_hi=10, s_min=90, v_min=70),
    "gray": dict(s_max=125, v_min=50, v_max=130),
}

# 每个类别的实例形状约束：面积、边长、填充率
# 这些值是在 1108_All 全部 36 集上网格搜索 + 奇偶留出验证出来的（三项均 100%/0%）：
#   · 边长上限必须放到 62 —— 灰方块常被斜放，包围盒会胀到 55×55，卡 48 会整块漏检（ep23）
#   · 灰 V 下限必须降到 50 —— 进了黄板的方块 V 只有 62-85，卡 60 会漏（ep18/26/27/29/30）
#   · 面积上限 1900 是卡机械臂橙色外壳的 —— 它面积 3581-8518，长边 72-137
#   · 填充率下限 0.45 是卡阴影散斑的 —— 阴影连通域填充率只有 0.17-0.44
SHAPE = {
    "red":  dict(area=(1100, 1900), side=(34, 58), fill=(0.55, 0.90)),
    "gray": dict(area=(800, 1900), side=(34, 62), fill=(0.45, 0.97)),
}


def mask(img, kind):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    t = THR[kind]
    if kind == "red":
        m = (((H <= t["h_hi"]) | (H >= t["h_lo"])) & (S > t["s_min"]) & (V > t["v_min"]))
    else:
        m = ((S < t["s_max"]) & (V >= t["v_min"]) & (V <= t["v_max"]))
    return (m.astype(np.uint8) * 255)


def detect(img, kind, roi=None, min_area=None):
    """返回实例列表 [{box, area, fill, cx, cy}]。roi=(x,y,w,h) 时只在该框内找。"""
    m = mask(img, kind)
    if roi is not None:
        x, y, w, h = (int(v) for v in roi)
        keep = np.zeros_like(m)
        keep[y:y + h, x:x + w] = m[y:y + h, x:x + w]
        m = keep
    cfg = SHAPE[kind]
    lo, hi = cfg["area"] if min_area is None else (min_area, cfg["area"][1])
    slo, shi = cfg["side"]
    flo, fhi = cfg["fill"]
    H, W = m.shape
    out = []
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        a = int(st[i, 4])
        x, y, w, h = (int(v) for v in st[i, :4])
        if not (lo <= a <= hi):
            continue
        if not (slo <= w <= shi and slo <= h <= shi):
            continue
        fill = a / max(1, w * h)
        if not (flo <= fill <= fhi):
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue                      # 贴画面边：是板沿/桌沿，不是方块
        out.append(dict(box=[x, y, w, h], area=a, fill=round(fill, 3),
                        cx=x + w // 2, cy=y + h // 2))
    out.sort(key=lambda d: -d["area"])
    return out


def count(img, kind, roi=None):
    return len(detect(img, kind, roi))


def nearest(img, kind, point, roi=None):
    """离 point 最近的实例；没有则 None。用于"抓离手最近的那块"。"""
    cands = detect(img, kind, roi)
    if not cands:
        return None
    px, py = point
    return min(cands, key=lambda d: (d["cx"] - px) ** 2 + (d["cy"] - py) ** 2)


def center_of(img, kind, roi=None):
    cands = detect(img, kind, roi)
    if not cands:
        return None
    d = cands[0]
    return (d["cx"], d["cy"])
