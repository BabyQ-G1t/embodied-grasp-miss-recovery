#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_ready.py —— 开工前的环境体检。装好机械臂、录完数据之后，先跑这一条。

它不动机器人，只回答六个问题：
  1. LeLab 服务起了没
  2. 从臂插上了没（串口在不在）
  3. 几台相机能用，分别是哪个 index
  4. 数据集在哪、每个有几条 episode
  5. 有没有训好的模型可以跑
  6. 当前画面里的方块，判定器认不认得出来

跑法：
    python check_ready.py
    python check_ready.py --gray 30,150,90

输出一份体检报告，认不出的地方会直接告诉你怎么调。
"""

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 关掉打不开相机时的刷屏警告
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import cv2
import numpy as np
import urllib.request
import urllib.error

try:
    cv2.setLogLevel(0)
except Exception:
    pass

HOST = "http://127.0.0.1:8000"
CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"
OUT = pathlib.Path(__file__).parent / "check_out"
BIN_JSON = pathlib.Path(__file__).parent / "bin_roi.json"

# 显式禁用系统代理：localhost 经代理会返回 502，误判成 LeLab 没起来
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

OK, BAD, WARN = "[OK]", "[X] ", "[!] "


def api(path, timeout=3):
    try:
        with _OPENER.open(HOST + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace") or "{}")
    except urllib.error.HTTPError as e:
        return {"_error": e.code}
    except Exception as e:
        return {"_error": str(e)}


def probe_cameras(max_idx=4):
    """返回 [(index, (w,h))]，能打开的才在里面。"""
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, f = cap.read()
            if ok and f is not None:
                h, w = f.shape[:2]
                found.append((i, (w, h)))
            else:
                found.append((i, None))
        cap.release()
    return found


def grab(index, warm=6):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    f = None
    for _ in range(warm):
        ok, tmp = cap.read()
        if ok:
            f = tmp
    cap.release()
    return f


def imwrite_u(path, img):
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if ok:
        buf.tofile(str(path))
    return ok


def mask_of(img, gray, hsv_range=None):
    """直接复用 run_trial 的判定实现，避免两边口径漂移。"""
    from run_trial import mask_of as _m
    return _m(img, gray, hsv_range)


def find_blocks(m, min_area=600, max_area=None):
    from run_trial import find_blocks as _f
    return _f(m, min_area)


def section(title):
    print(f"\n{'=' * 58}\n{title}\n{'=' * 58}")


def main():
    ap = argparse.ArgumentParser(description="开工前的环境体检")
    ap.add_argument("--gray", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=(30, 170, 100))
    ap.add_argument("--hsv", type=lambda s: [int(x) for x in s.split(",")],
                    # v_lo 140 / s_hi 255 与 run_trial 保持一致：灯光偏暗时块侧面只有
                    # V140–180，v_lo=180 会整块漏检（2026-11-01 实测 0 检出）。
                    default=(166, 120, 140, 12, 255, 255),
                    help="彩色方块范围，默认红色块；与 --gray 取并集。"
                         "h_lo>h_hi 表示色相绕回（166..179 + 0..12）")
    ap.add_argument("--target", type=int, default=1, help="桌上一共几个方块（本实验固定 1）")
    ap.add_argument("--cam", type=int, default=0, help="用哪个 index 拍判定用的照片")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    todo = []

    # 1. LeLab 服务
    section("1 / 6  LeLab 服务")
    h = api("/health")
    if h.get("_error"):
        print(f"{BAD}LeLab 没在跑（{str(h['_error'])[:60]}）")
        print("     -> 先启动 LeLab，浏览器能打开 localhost:8000 就算起来了")
        todo.append("启动 LeLab")
    else:
        print(f"{OK}LeLab 在跑：{h}")

    # 2. 串口
    section("2 / 6  从臂串口")
    port_file = CACHE / "ports" / "follower_port.txt"
    port = port_file.read_text(encoding="utf-8", errors="replace").strip() if port_file.exists() else None
    if port:
        print(f"{OK}记录的从臂串口：{port}")
        print("     （串口在不在，插上机械臂后用设备管理器核对一眼）")
    else:
        print(f"{WARN}没找到端口记录文件 {port_file}")
        todo.append("确认从臂插上、串口与 LeLab 里选的一致")

    # 3. 相机
    section("3 / 6  相机")
    cams = probe_cameras()
    if not cams:
        print(f"{BAD}一台都打不开")
        todo.append("检查相机 USB 是否插好；若 LeLab 正在占用，先停掉再测")
    else:
        for idx, wh in cams:
            print(f"{OK}index {idx}  分辨率 {wh[0]}x{wh[1]}" if wh else f"{WARN}index {idx} 能开但读不到画面")
        if len(cams) == 1:
            print(f"{WARN}只找到 1 台。策略通常需要 2 路（环境 + 腕部）")
            todo.append("确认第二台相机插好；或改策略只用一路")

        # --- 相机契约：模型侧的硬要求，不满足就白跑 ---
        # act_2026-11-05_11-52-59 的 input_features 里写死两个 VISUAL：
        #   observation.images.F1 / F2，shape (3, 480, 640)
        # LeLab 的 rollout 不传 --rename_map，相机字典的 key 会原样拼成
        # observation.images.<key>，所以 key 必须字面等于 F1 / F2。
        print(f"\n     模型要求：2 路，键名 F1 + F2（大小写敏感），分辨率 640x480")
        for idx, wh in cams:
            shot = grab(idx)
            if shot is None:
                continue
            dst = OUT / f"cam{idx}.png"
            if imwrite_u(dst, shot):
                print(f"     快照  index {idx} -> {dst}")
            if wh and wh != (640, 480):
                print(f"{WARN}index {idx} 是 {wh[0]}x{wh[1]}，不是 640x480；"
                      f"LeLab 会缩放到 640x480，画面会被拉伸")
                todo.append(f"把 index {idx} 的分辨率设成 640x480")
        print("     对号（打开上面两张快照）：")
        print("       F1 = 俯视固定机 —— 看得到整块蓝板 + 桌面，画面里没有夹爪指")
        print("       F2 = 腕部机    —— 画面下方有两个夹指，蓝板占画面大部分")
        print(f"{WARN}LeLab 推理页里 F1/F2 绑反了不会报错，但动作一定乱（抓空）。")
        print("       绑定后先在推理页看两路预览，确认哪一路有夹指再点执行。")
        print(f"{WARN}run_trial / check_ready 的 --cam 也必须是俯视那一路。")
        print("       拔插过 USB 后 Windows 的枚举顺序会变，index 可能互换，每次上机都要重看快照。")

    # 4. 数据集
    section("4 / 6  数据集")
    root = CACHE / "shuren2"
    total_ep = 0
    if root.exists():
        for info_p in sorted(root.glob("*/meta/info.json")):
            info = json.loads(info_p.read_bytes().decode("utf-8", errors="replace"))
            ep = info.get("total_episodes", 0)
            total_ep += ep
            fr = info.get("total_frames", 0)
            cams_n = [k.split(".", 2)[2] for k in info["features"] if k.startswith("observation.images.")]
            flag = OK if ep >= 10 else WARN
            print(f"{flag}{info_p.parent.parent.name:<26} {ep:>3} 条  {fr:>6} 帧  {cams_n}")
        print(f"\n     合计 {total_ep} 条 episode")
        if total_ep < 20:
            print(f"{WARN}示教条数偏少。计划里要 32 条（8 个摆位 × 4），现在还差 {max(0, 32 - total_ep)} 条")
            todo.append(f"补录示教到 32 条（当前 {total_ep}）")
    else:
        print(f"{BAD}没找到 {root}")
        todo.append("确认数据集录在哪个目录")

    # 5. 模型
    section("5 / 6  训好的模型")
    jobs = api("/jobs")
    items = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
    if items:
        print(f"{OK}/jobs 里有 {len(items)} 个任务")
        if h.get("_error"):
            print(f"{WARN}LeLab 没起，checkpoints 查不到")
        else:
            jid = items[0].get("id")
            ck = api(f"/jobs/{jid}/checkpoints")
            print(f"     最近任务 {jid} 的 checkpoint：{str(ck)[:200]}")
    else:
        print(f"{BAD}/jobs 里没有训练任务（或服务未连接）")
        todo.append("先训一个 ACT 模型，否则没有策略可跑")
    train_dir = CACHE / "outputs" / "train"
    n_out = len(list(train_dir.iterdir())) if train_dir.exists() else 0
    print(f"{'[OK]' if n_out else '[!] '}本地训练输出目录有 {n_out} 项（{train_dir}）")

    # 6. 判定器
    section("6 / 6  判定器在当前画面上认不认得方块")
    if not cams:
        print(f"{WARN}没有可用相机，跳过")
    else:
        f = grab(args.cam)
        if f is None:
            print(f"{BAD}index {args.cam} 拍不到帧，本次没有生成新截图")
            print("     -> 图片还是上一次的，别看它判断现场")
            print("     -> 常见原因：LeLab 正占用相机，先停掉 LeLab 再跑")
            todo.append(f"相机 index {args.cam} 拍不到帧")
        else:
            from run_trial import overlap_ratio, px_in_box, unit_px_of
            roi = None
            if BIN_JSON.exists():
                try:
                    roi = tuple(int(v) for v in json.loads(
                        BIN_JSON.read_text(encoding="utf-8"))["bin"])
                except Exception:
                    roi = None

            m = mask_of(f, args.gray, args.hsv)
            bl = find_blocks(m)
            table = [b for b in bl if roi is None or overlap_ratio(b, roi) < 0.5]
            inside = [b for b in bl if roi is not None and overlap_ratio(b, roi) >= 0.5]

            vis = f.copy()
            for (x, y, w, h) in table:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (x, y, w, h) in inside:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 160, 255), 2)
            if roi is not None:
                X, Y, W2, H2 = roi
                cv2.rectangle(vis, (X, Y), (X + W2, Y + H2), (0, 0, 255), 2)
            p = OUT / f"cam{args.cam}_snapshot.png"
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"shot {stamp}  table={len(table)} bin={len(inside)} target={args.target}"
            cv2.rectangle(vis, (0, 0), (640, 24), (32, 32, 32), -1)
            cv2.putText(vis, line, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            imwrite_u(p, vis)
            try:
                st = p.stat()
                print(f"     截图已重写：{p}")
                print(f"       写入时间 {time.strftime('%H:%M:%S', time.localtime(st.st_mtime))}"
                      f"  大小 {st.st_size} B  （每次运行都覆盖，看图上角的时间戳确认是新的）")
            except OSError as exc:
                print(f"{BAD}截图写入失败：{p}（{exc}）")
                todo.append("截图写入失败")

            print(f"     桌面认出 {len(table)} 个（绿框），收纳区内 {len(inside)} 个（橙框）"
                  f"，目标 {args.target} 个")
            for (x, y, w, h) in table:
                print(f"       桌面 位置({x},{y}) {w}x{h}px")
            if len(table) < args.target:
                print(f"{BAD}桌面方块不足：要 {args.target} 个，只认出 {len(table)} 个")
                only_gray = [b for b in find_blocks(mask_of(f, args.gray, None))
                             if roi is None or overlap_ratio(b, roi) < 0.5]
                only_red = [b for b in find_blocks(mask_of(f, None, args.hsv))
                            if roi is None or overlap_ratio(b, roi) < 0.5]
                print(f"       拆分：深色掩膜认出 {len(only_gray)} 个，彩色掩膜认出 {len(only_red)} 个")
                if len(only_gray) < 1:
                    print("     -> 深色块漏了：放宽 --gray，如 --gray 20,190,110")
                if len(only_red) < 1:
                    print("     -> 彩色块漏了：放宽 --hsv，如 --hsv 0,110,150,12,220,255")
                    print("        (红块若偏暗就降 v_lo；若偏淡就降 s_lo；色相偏移就加 h_hi)")
                todo.append(f"桌面方块不足（{len(table)}/{args.target}）")
            else:
                print(f"{OK}桌面方块数量达标")

            if roi is None:
                print(f"{BAD}没有 bin_roi.json，跑不了正式试验")
                todo.append("运行 calibrate_bin.bat 标定收纳区")
            elif inside:
                unit = unit_px_of(m, table) if table else 0
                bin_px = px_in_box(m, roi)
                print(f"{BAD}收纳区里还有东西：{bin_px} px"
                      + (f"，约 {bin_px / unit:.1f} 块当量" if unit else ""))
                todo.append("清空收纳区（实验以“进收纳区的块数”计成功）")
            else:
                print(f"{OK}收纳区是空的（框 {roi}）")

    # 汇总
    section("体检结论")
    if not todo:
        print(f"{OK}六项都过了，可以直接开始跑实验。")
        print("\n下一步：双击 run_trial_local.bat，输入分组（调试用 B，正式 60 次用 MIX）")
    else:
        print(f"{WARN}还有 {len(todo)} 件事要先处理：")
        for i, t in enumerate(todo, 1):
            print(f"   {i}. {t}")
        print("\n处理后重跑这个脚本，全绿了再往下走。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
