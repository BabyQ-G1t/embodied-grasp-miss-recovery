# -*- coding: utf-8 -*-
"""全量检测「方块是否被一次抓起 / 是否中途掉落」。

方法
----
1. 每帧在「两块垫板外括 60px」范围内找所有合格的深灰连通块（几何约束排除桌面暗色长条与画面右侧白盒）。
2. 以首帧落在蓝板内的最大块为种子，逐帧最近邻跟踪（门限 90px，短暂丢失后放宽到 165px）。
3. 判据：
   D1 夹爪张开段数 != 2                            -> 多次开合（重抓）
   D2 搬运段内 cy 出现「高位<170 -> 低位>275」回落   -> 抬起后又落回板面
   D3 搬运段内方块停在垫板外(OUT)且 cy>285 连续>=4帧 -> 掉在桌面
   D4 末帧方块不在黄板内                            -> 放置失败

输出 scan_1108/drop_summary.csv + drop_tracks.json（逐帧轨迹）
用法: python drop_scan.py [--step 2] [--eps 7] [--only opq1108-1]
"""
import argparse
import csv
import json
import pathlib

import av
import cv2
import numpy as np
import pandas as pd

import scan_1108 as S

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "scan_1108"
PKL = OUT / "drop_tracks.json"

GEOM = dict(wmin=20, wmax=95, hmin=20, hmax=95, ar=(0.5, 2.0), amin=500)
OPEN_THR = 22.0
ROI_PAD = 60
NEAR, NEAR_RELAX, LOST_RELAX = 90.0, 165.0, 2


def candidates(img, mats):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 2] < S.GRAY_V_MAX) & (hsv[:, :, 1] < S.GRAY_S_MAX)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if mats:
        xs, ys = [], []
        for (x, y, w, h) in mats.values():
            xs += [x, x + w]
            ys += [y, y + h]
        roi = np.zeros_like(m)
        roi[max(0, min(ys) - 30):img.shape[0],
            max(0, min(xs) - ROI_PAD):min(img.shape[1], max(xs) + ROI_PAD)] = 255
        m = cv2.bitwise_and(m, roi)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < GEOM["amin"]:
            continue
        if not (GEOM["wmin"] <= w <= GEOM["wmax"] and GEOM["hmin"] <= h <= GEOM["hmax"]):
            continue
        if not (GEOM["ar"][0] <= w / max(1, h) <= GEOM["ar"][1]):
            continue
        cx, cy = x + w // 2, y + h // 2
        out.append({"cx": cx, "cy": cy, "w": w, "h": h, "area": a,
                    "pos": S.where({"cx": cx, "cy": cy}, mats)})
    return out


def track(ds, ep, step=2):
    base = S.ROOT / ds
    meta = pd.read_parquet(base / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    row = meta[meta.episode_index == ep].iloc[0]
    data = pd.read_parquet(base / "data" / "chunk-000" / "file-000.parquet")
    st6 = np.stack(data[data.episode_index == ep]["observation.state"].values).astype(float)
    grip = st6[:, 5]

    t0 = float(row["videos/observation.images.F1/from_timestamp"])
    t1 = float(row["videos/observation.images.F1/to_timestamp"])
    fi = int(row["videos/observation.images.F1/file_index"])
    c = av.open(str(base / "videos" / "observation.images.F1" / "chunk-000" / f"file-{fi:03d}.mp4"))
    st = c.streams.video[0]
    c.seek(int(max(0, t0) * 1e6), backward=True)

    recs, k, last, lost = [], 0, None, 0
    for fr in c.decode(st):
        ts = float(fr.pts * st.time_base)
        if ts < t0 - 1e-3:
            continue
        if ts > t1 + 1e-3:
            break
        if k % step == 0:
            img = fr.to_ndarray(format="bgr24")
            d = S.detect(img)
            cands = candidates(img, d["mats"])
            pick = None
            if last is None:
                pool = [q for q in cands if q["pos"] == "blue"] or cands
                if pool:
                    pick = max(pool, key=lambda q: q["area"])
            else:
                lim = NEAR_RELAX if lost >= LOST_RELAX else NEAR
                near = [q for q in cands
                        if np.hypot(q["cx"] - last[0], q["cy"] - last[1]) <= lim]
                if near:
                    pick = min(near, key=lambda q: np.hypot(q["cx"] - last[0], q["cy"] - last[1]))
            if pick:
                last, lost = (pick["cx"], pick["cy"]), 0
            else:
                lost += 1
            af = int(round((ts - t0) / max(1e-6, t1 - t0) * (len(grip) - 1)))
            af = min(max(af, 0), len(grip) - 1)
            recs.append({"ts": round(ts - t0, 3), "af": af,
                         "grip": round(float(grip[af]), 1),
                         "gx": pick["cx"] if pick else None,
                         "gy": pick["cy"] if pick else None,
                         "pos": pick["pos"] if pick else "LOST",
                         "lost": lost,
                         "yellow": d["mats"].get("yellow"),
                         "blue": d["mats"].get("blue")})
        k += 1
    c.close()
    return recs, grip


def open_segs(grip, thr=OPEN_THR):
    closed = grip < thr
    segs, ins, stt = [], False, 0
    for i, cc in enumerate(closed):
        if not cc and not ins:
            stt, ins = i, True
        elif cc and ins:
            segs.append([stt, i])
            ins = False
    if ins:
        segs.append([stt, len(closed)])
    return segs


def analyze(recs, grip, ds="", ep=-1):
    segs = open_segs(grip)
    a, b = (segs[0][1], segs[-1][0]) if len(segs) >= 2 else (0, len(recs) - 1)
    carry = [r for r in recs if a <= r["af"] <= b and r["gy"] is not None]
    ys = [(r["gy"], r["pos"]) for r in carry]

    fall_big = 0
    for i in range(1, len(ys)):
        if ys[i][0] - ys[i - 1][0] > 25 and ys[i - 1][0] < 170 and ys[i][0] > 275:
            fall_big += 1

    run = best_out = 0
    for _, p in ys:
        if p == "OUT":
            run += 1
            best_out = max(best_out, run)
        else:
            run = 0

    yv = [y[0] for y in ys]
    tail = [r for r in recs if r["gy"] is not None][-3:]
    end = tail[-1] if tail else None
    end_pos, end_margin = (end["pos"], None) if end else ("NONE", None)
    if end and end["yellow"]:
        x, y, w, h = end["yellow"]
        end_margin = round(float(min(end["gx"] - x, x + w - end["gx"],
                                     end["gy"] - y, y + h - end["gy"])), 1)

    flags = []
    if len(segs) != 2:
        flags.append(f"D1_张开{len(segs)}段")
    if fall_big:
        flags.append(f"D2_落回{fall_big}次")
    if best_out >= 4:
        flags.append(f"D3_板外低位{best_out}帧")
    if end_pos != "yellow":
        flags.append(f"D4_末{end_pos}")

    return {"ds": ds.split("_")[0], "ep": ep, "n_open": len(segs),
            "carry_n": len(carry), "y_min": min(yv) if yv else None,
            "y_max": max(yv) if yv else None,
            "fall_big": fall_big, "out_low": best_out,
            "lost_frames": sum(1 for r in carry if r["lost"] > 0),
            "end_pos": end_pos, "end_margin": end_margin,
            "flags": ";".join(flags)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--eps", default=None)
    ap.add_argument("--only", default=None, help="数据集名前缀，如 opq1108-1")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    todos = []
    for ds in S.DS:
        if args.only and not ds.startswith(args.only):
            continue
        meta = pd.read_parquet(S.ROOT / ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        eps = sorted(int(x) for x in meta.episode_index)
        if args.eps:
            want = {int(x) for x in args.eps.split(",")}
            eps = [e for e in eps if e in want]
        todos += [(ds, e) for e in eps]

    rows, tracks = [], {}
    for i, (ds, ep) in enumerate(todos, 1):
        recs, grip = track(ds, ep, args.step)
        res = analyze(recs, grip, ds, ep)
        rows.append(res)
        tracks[f"{ds}|{ep}"] = {"summary": res,
                                "ys": [[r["af"], r["gy"], r["gx"], r["pos"]]
                                       for r in recs if r["gy"] is not None]}
        if not args.quiet:
            print(f"[{i:>2}/{len(todos)}] {res['ds']:9s} ep{ep:<3d} 张开{res['n_open']}段 "
                  f"y {res['y_min']}~{res['y_max']} 落回{res['fall_big']} 板外{res['out_low']} "
                  f"末{res['end_pos']}({res['end_margin']}) {res['flags']}", flush=True)

    PKL.write_text(json.dumps(tracks, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "drop_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["flags"]]
    print("\n" + "=" * 70)
    print(f"共 {len(rows)} 集，命中标记 {len(bad)} 集")
    for r in bad:
        print(f"  {r['ds']} ep{r['ep']}  {r['flags']}   (y {r['y_min']}~{r['y_max']}, 末 {r['end_pos']})")
    n1 = sum(1 for r in rows if r["n_open"] == 2)
    print(f"张开段数：2段 {n1} 集 / 其它 {len(rows) - n1} 集")


if __name__ == "__main__":
    main()
