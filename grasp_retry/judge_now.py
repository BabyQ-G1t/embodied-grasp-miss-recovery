"""早停后补判：机械臂停住后立刻取一帧，用与 run_trial 完全相同的双证据判据
（原位是否变空 + 收纳区是否增料）回答「这条指令到底完成没有」。

为什么需要它：
    run_trial 的判定发生在「一轮跑完」之后（f_post vs f_pre）。用户在轮次中途点早停，
    进程被 taskkill，判定那一步永远不会执行，页面只能报「未判定」。
    本脚本把这一步提前做掉：拿本轮开始前已经落盘的那一帧当基准（t0 或 a{n-1}），
    现在补拍一张收尾帧，直接套 run_trial 的 mask_of / check_boxes / judge，
    门限与计量口径完全一致，不另立一套标准。

用法：
    python judge_now.py --trial 7 --pos P1 --round 1 --cam 0 --json
输出（--json 只打印一行 JSON）：
    {"ok": true, "verdict": "SUCCESS|EMPTY|LOST|UNKNOWN", "reason": "...",
     "left_ratio": 0.12, "bin_gain": 1420, "unit_px": 1500.0, ...}
退出码：0 判出结果（含 EMPTY/LOST）；2 无法判定（相机/基准帧/当量有问题）。
"""

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_trial as rt   # 复用同一套判据，禁止在这里另写门限

HERE = pathlib.Path(__file__).resolve().parent
SUCCESS, EMPTY, LOST, UNKNOWN = "SUCCESS", "EMPTY", "LOST", "UNKNOWN"


def imread_u(path):
    """Windows 中文路径下 cv2.imread 会静默返回 None，必须走 imdecode。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if not data.size:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def pre_frame_path(outdir, pos, trial, rnd):
    """本轮开始前的那一帧：第 1 轮是 t0，之后是上一轮的收尾帧 a{n-1}。"""
    if rnd <= 1:
        return outdir / f"{pos}_{trial:02d}_t0.png"
    return outdir / f"{pos}_{trial:02d}_a{rnd - 1}.png"


def main():
    ap = argparse.ArgumentParser(description="早停后补做本轮判定")
    ap.add_argument("--trial", type=int, required=True, help="本轮试验编号（trials.csv 的「试验」列）")
    ap.add_argument("--pos", default="P1")
    ap.add_argument("--round", type=int, default=1, help="停在第几轮（决定用哪一帧当基准）")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--outdir", default="trial_logs")
    ap.add_argument("--target", type=int, default=1, help="桌上方块数（本实验固定 1 块）")
    ap.add_argument("--bin", type=lambda s: tuple(int(x) for x in s.split(",")), default=None,
                    help="收纳区 ROI x,y,w,h；不给则读 bin_roi.json")
    ap.add_argument("--gray", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=(30, 170, 100))
    ap.add_argument("--dark-v", type=int, default=100,
                    help="V<=此值即认作方块（灰块进黄板后 S 会被浸染成 190，只靠 -gray 会漏）")
    ap.add_argument("--hsv", type=lambda s: [int(x) for x in s.split(",")],
                    default=None,
                    help="彩色方块掩膜，默认关闭；1108 布局桌上那颗红圆柱是干扰物，"
                         "开起来会被当候选块（P1_14 误判 EMPTY 的直接原因）")
    ap.add_argument("--min-area", type=int, default=600)
    ap.add_argument("--keep-ratio", type=float, default=0.40)
    ap.add_argument("--settle", type=float, default=1.5, help="停机后等机械臂完全静止的秒数")
    ap.add_argument("--json", action="store_true", help="只输出一行 JSON，供总控解析")
    a = ap.parse_args()

    outdir = HERE / a.outdir
    out = {"ok": False, "verdict": UNKNOWN, "reason": "", "trial": a.trial,
           "round": a.round, "pos": a.pos, "left_ratio": None, "bin_gain": None,
           "bin_px_pre": None, "bin_px_post": None, "unit_px": None,
           "pre_frame": None, "post_frame": None}

    def done(code=0, reason="", verdict=None):
        out["reason"] = reason
        if verdict:
            out["verdict"] = verdict
            out["ok"] = verdict != UNKNOWN
        if a.json:
            print(json.dumps(out, ensure_ascii=False))
        else:
            print(f"[补判] {out['verdict']}｜{reason}")
        return code

    try:
        roi = a.bin or rt.load_bin_roi(None)
    except Exception as e:
        return done(2, f"收纳区 ROI 读不到：{e}")
    if not roi:
        return done(2, "没有收纳区 ROI（先跑 calibrate_bin 或用 --bin 指定）")

    pre_p = pre_frame_path(outdir, a.pos, a.trial, a.round)
    out["pre_frame"] = pre_p.name
    f_pre = imread_u(pre_p)
    if f_pre is None:
        return done(2, f"找不到本轮开始前的基准帧 {pre_p.name}，无法补判")

    # 停机后机械臂可能还在余动，先等它停稳再拍，否则拍到运动模糊帧会误判「已离开原位」
    if a.settle > 0:
        time.sleep(a.settle)
    f_post = rt.open_cam(a.cam)
    if f_post is None:
        return done(2, f"相机 {a.cam} 打不开（可能仍被 LeLab 或预览页面占着）")
    okf, why = rt.frame_quality(f_post)
    if not okf:
        return done(2, f"收尾帧不可用：{why}")
    post_p = outdir / f"{a.pos}_{a.trial:02d}_stopjudge.png"
    try:
        rt.imwrite_u(post_p, f_post)
        out["post_frame"] = post_p.name
    except Exception:
        pass

    m0 = rt.mask_of(f_pre, a.gray, a.hsv, a.dark_v)
    # 原位框要从"开局帧"取（方块原始位置），不能用本轮开始前的帧——
    # 那一帧里方块可能已经被上一轮拿走了，框会画到空桌面上。
    t0_p = outdir / f"{a.pos}_{a.trial:02d}_t0.png"
    f_t0 = imread_u(t0_p)
    m_base = rt.mask_of(f_t0, a.gray, a.hsv, a.dark_v) if f_t0 is not None else m0
    out["base_frame"] = t0_p.name if f_t0 is not None else pre_p.name
    # 与 run_trial 同一套选块逻辑：踢掉收纳区里的、踢掉两块板范围外的（上沿机械臂）
    cands, boxes = rt.target_boxes(m_base, roi, rt.pick_roi_of(), a.target, a.min_area)
    if not boxes:
        return done(2, "开局帧里没找到原位方块框，无法补判")
    unit_px = rt.unit_px_of(m_base, boxes)
    out["unit_px"] = round(unit_px, 1)
    if unit_px <= 0:
        return done(2, "单块当量算不出来（开局帧框内没有方块色像素）")

    # 本轮开始时还留在原位的方块：拿"本轮开始前帧"与开局帧比，
    # 上几轮已经拿走的块不参与本轮判定（合 run_trial 里 alive 的语义）。
    alive = []
    for b in boxes:
        r_base = rt.ratio_in_box(m_base, b)
        r_pre = rt.ratio_in_box(m0, b)
        if r_base > 1e-6 and r_pre / r_base >= a.keep_ratio:
            alive.append(b)
    if not alive:
        return done(0, "本轮开始时原位已空，无方块可判", LOST)

    m_post = rt.mask_of(f_post, a.gray, a.hsv, a.dark_v)
    bin_pre = rt.px_in_box(m0, roi)
    bin_post = rt.px_in_box(m_post, roi)
    out.update(bin_px_pre=bin_pre, bin_px_post=bin_post, bin_gain=bin_post - bin_pre)

    status = rt.check_boxes(m0, m_post, alive, a.keep_ratio)
    out["left_ratio"] = round(min(s["rel"] for s in status), 3) if status else None
    v = rt.judge(status, bin_post - bin_pre, unit_px)
    detail = (f"原位占比 {out['left_ratio']:.2f}（<{a.keep_ratio} 算已离开），"
              f"收纳区 {bin_pre}→{bin_post} px（增 {bin_post - bin_pre:+d}，"
              f"门限 {rt.BIN_GAIN_UNIT * unit_px:.0f}）")
    if v == SUCCESS:
        return done(0, f"原位已空且收纳区增料，判为已入区；{detail}", SUCCESS)
    if v == EMPTY:
        return done(0, f"原位仍有方块，说明没抓到；{detail}", EMPTY)
    return done(0, f"原位已空但收纳区没增料（推走／掉落／遮挡）；{detail}", LOST)


if __name__ == "__main__":
    sys.exit(main())
