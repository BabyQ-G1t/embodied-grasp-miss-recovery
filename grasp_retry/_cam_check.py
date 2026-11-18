# -*- coding: utf-8 -*-
"""相机视角 + 曝光 + 摆位三项检查。

训练基准（4 个数据集首帧实测，彼此差异 <2px、亮度 140–162）：
    蓝板 bbox     = (253, 54, 191, 250)
    画面平均亮度  = 148（中位）
    过曝像素占比  < 5%
    块中心        = 1026 位 (221,195) / 1028 位 (456,270)

用法：python _cam_check.py
"""
import os
import sys
import pathlib

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _pos_audit import blocks_of

REF_BOX = (253, 54, 191, 250)
REF_MEAN = 148.7
REF_PTS = [("1025", 468, 210), ("1026", 221, 195), ("1027", 218, 126), ("1028", 456, 270)]
TOL_BOX = 6
TOL_PT = 31
TOL_MEAN = 8


def blue_box(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (85, 80, 60), (125, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    box = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return box if box[2] * box[3] > 8000 else None


def grab(i, warm=30):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    best, bm = None, -1.0
    for _ in range(warm):
        ok, f = cap.read()
        if ok and f is not None:
            m = float(f.mean())
            if m > bm:
                bm, best = m, f
    cap.release()
    return best


def scan(indices=range(5)):
    """扫全部索引，报告亮度/过曝/蓝板，挑最像环境相机的一台。

    选法不是"最亮"，而是"能检出蓝板 + 亮度最接近训练值"——
    过曝到 83% 时画面全白，蓝板反而检不出，越亮越糟。
    """
    rows = []
    for i in indices:
        f = grab(i)
        if f is None:
            rows.append(dict(idx=i, frame=None, mean=0.0, clip=0.0, bb=None))
            continue
        v = f.max(axis=2)
        rows.append(dict(idx=i, frame=f, mean=float(f.mean()),
                         clip=float((v >= 250).mean() * 100), bb=blue_box(f)))
        ok, buf = cv2.imencode(".png", f)
        (HERE / "check_out").mkdir(exist_ok=True)
        (HERE / "check_out" / f"cam{i}_probe.png").write_bytes(buf.tobytes())
    print("扫描相机：")
    for r in rows:
        if r["frame"] is None:
            print(f"  index {r['idx']}: 打不开")
        else:
            print(f"  index {r['idx']}: 亮度 {r['mean']:>6.1f}  过曝 {r['clip']:>5.1f}%  "
                  + (f"蓝板 {r['bb']}" if r["bb"] else "蓝板 未检出"))
    cands = [r for r in rows if r["bb"] is not None]
    if cands:
        pick = min(cands, key=lambda r: abs(r["mean"] - REF_MEAN))
        print(f"  -> 环境相机 = index {pick['idx']}（能检出蓝板，亮度最接近基准）")
        return pick
    live = [r for r in rows if r["frame"] is not None]
    if live:
        pick = min(live, key=lambda r: abs(r["mean"] - REF_MEAN))
        print(f"  -> 没有任何一路能检出蓝板，暂取 index {pick['idx']}")
        return pick
    return None


def main():
    pick = scan()
    if pick is None:
        print("[X] 一路相机都拿不到画面")
        return 3
    img = pick["frame"]
    ok_all = True
    print(f"\n采用 index {pick['idx']}  {img.shape[1]}x{img.shape[0]}")

    print("\n== 1. 曝光 ==")
    print(f"  训练基准 亮度 {REF_MEAN:.0f}±{TOL_MEAN}，过曝 <5%")
    print(f"  当前     亮度 {pick['mean']:.1f}，过曝 {pick['clip']:.1f}%")
    if abs(pick["mean"] - REF_MEAN) <= TOL_MEAN and pick["clip"] < 5:
        print("  [OK] 曝光正常")
    else:
        ok_all = False
        if pick["mean"] > REF_MEAN + TOL_MEAN:
            print("  [X] 画面过亮 —— 减少射向桌面的直射光（台灯挪远/调暗/关顶灯），"
                  "或把相机曝光调低");
        else:
            print("  [X] 画面偏暗 —— 增加光照")
        print("      过曝时画面白化，模型看到的等于空白纸，抓空是必然")

    print("\n== 2. 相机视角（对齐蓝板）==")
    bb = pick["bb"]
    if bb is None:
        print("  [X] 蓝板检不出（多半是过曝把蓝色也冲白了），先把曝光调回来再看")
        ok_all = False
    else:
        d = [bb[i] - REF_BOX[i] for i in range(4)]
        print(f"  训练基准 x={REF_BOX[0]} y={REF_BOX[1]} w={REF_BOX[2]} h={REF_BOX[3]}")
        print(f"  当前     x={bb[0]} y={bb[1]} w={bb[2]} h={bb[3]}")
        print(f"  偏差     dx={d[0]:+d} dy={d[1]:+d} dw={d[2]:+d} dh={d[3]:+d}（容差 ±{TOL_BOX}）")
        if all(abs(v) <= TOL_BOX for v in d):
            print("  [OK] 视角与训练一致")
        else:
            ok_all = False
            tip = []
            if d[1] > TOL_BOX:
                tip.append(f"画面下沉 {d[1]}px，相机抬高/后拉")
            elif d[1] < -TOL_BOX:
                tip.append(f"画面高出 {-d[1]}px，相机压低/前移")
            if d[2] < -TOL_BOX:
                tip.append(f"蓝板窄 {-d[2]}px，相机拉远")
            elif d[2] > TOL_BOX:
                tip.append(f"蓝板宽 {d[2]}px，相机靠近")
            if abs(d[0]) > TOL_BOX:
                tip.append(("右" if d[0] < 0 else "左") + f"移 {abs(d[0])}px")
            print("  [X] " + "；".join(tip))

    print("\n== 3. 方块摆位 ==")
    bl = blocks_of(img)
    print(f"  检出 {len(bl)} 个（应为 2）")
    if len(bl) != 2:
        ok_all = False
    for i, (bx, by, bw, bh) in enumerate(bl, 1):
        cx, cy = bx + bw // 2, by + bh // 2
        if not REF_PTS:
            continue
        d = sorted((np.hypot(cx - tx, cy - ty), nm, tx, ty) for nm, tx, ty in REF_PTS)[0]
        good = d[0] <= TOL_PT
        ok_all &= good
        print(f"  块{i} ({cx},{cy}) 最近 {d[1]}位({d[2]},{d[3]}) 距离 {d[0]:.0f}px -> "
              + ("OK" if good else "FAR"))

    vis = img.copy()
    if bb:
        cv2.rectangle(vis, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (255, 255, 0), 2)
    x, y, w, h = REF_BOX
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 1)
    for nm, tx, ty in REF_PTS:
        cv2.drawMarker(vis, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
        cv2.putText(vis, nm, (tx + 9, ty - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    for i, (bx, by, bw, bh) in enumerate(bl, 1):
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(vis, f"B{i}", (bx, max(12, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"mean={pick['mean']:.0f} clip={pick['clip']:.0f}%", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", vis)
    (HERE / "check_out" / "cam_check.png").write_bytes(buf.tobytes())

    print("\n" + "=" * 46)
    print("结论：" + ("三项都过，可以开跑" if ok_all else "未达标，按上面提示逐项调"))
    print(f"标注图：{HERE / 'check_out' / 'cam_check.png'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
