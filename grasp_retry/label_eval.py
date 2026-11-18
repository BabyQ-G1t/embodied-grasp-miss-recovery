#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_eval.py —— 判别器的正反样本评估（离线，不占真机）

为什么必须有这个脚本：
    只用"成功示教"的视频无法衡量判别器，因为成功样本里根本没有抓空。
    全部判成 SUCCESS 也能在 120 条成功样本上取得 100% 的"准确率"。
    要回答"判别器能不能发现失败"，必须采集并人工标注足量的负样本，
    并按类别分别报告检出率、漏检率与误报率，而不是一个笼统的准确率。

四类真值（labels.csv 的「真值」列）：
    SUCCESS  方块离开原位且确实进了收纳区
    EMPTY    策略执行了但没抓到，方块原位未动（含"机械臂遮挡原位"这一难例）
    LOST     方块离开原位却没进收纳区（被推走、掉到桌外）
    INVALID  该帧不可用（全黑、过曝、无纹理、相机未就绪）

样本要求（方案门禁的组成部分，缺一类即视为未通过）：
    SUCCESS  ≥ 40 条      EMPTY ≥ 40 条（其中遮挡型 ≥ 10）
    LOST     ≥ 10 条      INVALID ≥ 20 条
    样本须来自正式机位、正式道具、正式推理流程；不得从训练集里挑，
    且与示教数据不重叠（同一次执行不得同时出现在标定集与评估集）。

三个命令：
    python label_eval.py template --out labels.csv
        生成标注表模板（含示例行与字段说明）。
    python label_eval.py pair --video a.mp4 --id S001 --outdir pairs
        从一段录像截取"执行前/执行后"两帧，写入 pairs/S001_pre.png 与 _post.png，
        并追加一行到 labels.csv，真值留给人工填。
    python label_eval.py eval --labels labels.csv
        逐条跑判别，输出混淆矩阵与分项指标，并给出门禁判定。

判定与主执行器共用同一份实现（直接 import run_trial），
不另写一套阈值逻辑，避免"离线算得准、上机另一套"的两套口径问题。
"""

import argparse
import csv
import math
import pathlib
import sys

import cv2

import run_trial as R

# ── 门禁阈值（与方案「上机前门禁」一致，改动须同步方案）────────────────
GATE = dict(
    tpr_empty=0.90,        # 抓空检出率下限
    tpr_empty_lo=0.80,     # 抓空检出率 95% 置信区间下限
    far_success=0.05,      # 成功样本被误判为异常的比例上限
    far_success_hi=0.10,   # 同上，95% 置信区间上限
    tpr_lost=0.80,         # 推移/掉落检出率下限
    tpr_invalid=0.95,      # 取帧异常检出率下限
    n_success=40, n_empty=40, n_empty_occl=10, n_lost=10, n_invalid=20,
)

FIELDS = ["样本ID", "执行前图", "执行后图", "真值", "场景", "来源录像", "备注"]


# ======================================================================
# 统计工具
# ======================================================================

def wilson(k, n, z=1.96):
    """Wilson 置信区间。小样本下比正态近似可靠，避免"3/3=100%"这种假精确。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def pct(k, n):
    return f"{k}/{n} = {k / n:.1%}" if n else "—"


# ======================================================================
# 逐条判别：与主执行器同一套判定
# ======================================================================

def judge_pair(pre_path, post_path, bin_roi, target, gray, hsv, min_area, keep_ratio):
    """
    读一对图，跑与真机完全一致的判定流程。
    返回 (verdict, detail)。verdict ∈ SUCCESS/EMPTY/LOST/INVALID/SKIP。
    """
    f_pre = cv2.imread(str(pre_path))
    f_post = cv2.imread(str(post_path))
    if f_pre is None or f_post is None:
        return "INVALID", "图读不到"
    # 帧可用性门禁：先拦不可用帧，否则全黑帧会被当成"方块不在原位"
    for tag, f in (("执行前", f_pre), ("执行后", f_post)):
        ok, why = R.frame_quality(f)
        if not ok:
            return "INVALID", f"{tag}{why}"
    m_pre = R.mask_of(f_pre, gray, hsv)
    m_post = R.mask_of(f_post, gray, hsv)
    cands = R.find_blocks(m_pre, min_area)
    table = [b for b in cands if R.overlap_ratio(b, bin_roi) < 0.5]
    boxes = R.select_blocks(table, target)
    if not boxes:
        return "SKIP", "执行前帧识别不到方块（参数与机位不匹配）"
    unit_px = R.unit_px_of(m_pre, boxes)
    if unit_px <= 0:
        return "SKIP", "单块当量为 0"
    bin_gain = R.px_in_box(m_post, bin_roi) - R.px_in_box(m_pre, bin_roi)
    status = R.check_boxes(m_pre, m_post, boxes, keep_ratio)
    v = R.judge(status, bin_gain, unit_px)
    rels = ",".join(f"{s['rel']:.2f}" for s in status)
    return v, f"REL=[{rels}] gain={bin_gain}"


# ======================================================================
# 模板
# ======================================================================

TEMPLATE_ROWS = [
    ["S001", "pairs/S001_pre.png", "pairs/S001_post.png", "SUCCESS", "正常搬运", "", "方块确实进收纳区"],
    ["S002", "pairs/S002_pre.png", "pairs/S002_post.png", "SUCCESS", "正常搬运", "", ""],
    ["N001", "pairs/N001_pre.png", "pairs/N001_post.png", "EMPTY", "抓空", "", "策略夹空，方块原位未动"],
    ["N002", "pairs/N002_pre.png", "pairs/N002_post.png", "EMPTY", "抓空-遮挡", "", "执行后机械臂挡住原位"],
    ["N003", "pairs/N003_pre.png", "pairs/N003_post.png", "LOST", "推移", "", "方块被推到桌边，未进收纳区"],
    ["N004", "pairs/N004_pre.png", "pairs/N004_post.png", "LOST", "掉落", "", "方块掉出桌面"],
    ["N005", "pairs/N005_pre.png", "pairs/N005_post.png", "INVALID", "读帧异常", "", "相机被占用，全黑帧"],
]


def cmd_template(args):
    p = pathlib.Path(args.out)
    if p.exists() and not args.force:
        print(f"[!] {p} 已存在，未覆盖。要重建加 --force。")
        return 2
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(TEMPLATE_ROWS)
    print(f"已生成标注表模板：{p}")
    print("\n填写说明：")
    print("  样本ID   S 开头为成功样本，N 开头为负样本，递增编号")
    print("  执行前图 该轮执行【前】的一帧；执行后图 该轮执行【后】回待机的一帧")
    print("  真值     SUCCESS / EMPTY / LOST / INVALID 四选一")
    print("  场景     抓空、抓空-遮挡、推移、掉落、读帧异常、正常搬运")
    print(f"\n样本量要求：SUCCESS≥{GATE['n_success']}  EMPTY≥{GATE['n_empty']}"
          f"（其中遮挡≥{GATE['n_empty_occl']}）  LOST≥{GATE['n_lost']}"
          f"  INVALID≥{GATE['n_invalid']}")
    return 0


def cmd_pair(args):
    cap = cv2.VideoCapture(str(args.video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 10:
        cap.release()
        print(f"[!] {args.video} 帧数异常（{n}）。")
        return 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.pre_frame)
    ok1, pre = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.post_frame if args.post_frame >= 0 else n - 3)
    ok2, post = cap.read()
    cap.release()
    if not (ok1 and ok2):
        print("[!] 取帧失败。")
        return 2
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    R.imwrite_u(outdir / f"{args.id}_pre.png", pre)
    R.imwrite_u(outdir / f"{args.id}_post.png", post)
    line = [args.id, str(outdir / f"{args.id}_pre.png"), str(outdir / f"{args.id}_post.png"),
            args.truth or "", args.scene or "", str(args.video), ""]
    p = pathlib.Path(args.labels)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(FIELDS)
        w.writerow(line)
    print(f"已截取 {args.id}：{args.video}")
    print(f"  执行前 frames[{args.pre_frame}]  执行后 frames[{args.post_frame if args.post_frame >= 0 else n - 3}]")
    print(f"  已追加一行到 {p}（真值待人工核对）")
    return 0


# ======================================================================
# 评估
# ======================================================================

def cmd_eval(args):
    rows = list(csv.DictReader(open(args.labels, encoding="utf-8-sig")))
    if not rows:
        print("[!] 标注表是空的。")
        return 2
    bin_roi = R.load_bin_roi(args.bin)
    if bin_roi is None:
        print("[!] 缺收纳区 ROI。先跑 run_trial.py calibrate-bin，或给 --bin x,y,w,h。")
        return 2

    conf = {}          # (真值, 判定) -> 计数
    skipped = []
    done = 0
    for r in rows:
        truth = (r.get("真值") or "").strip().upper()
        if truth not in ("SUCCESS", "EMPTY", "LOST", "INVALID"):
            skipped.append((r.get("样本ID", "?"), "真值未填或非法"))
            continue
        v, detail = judge_pair(r["执行前图"], r["执行后图"], bin_roi, args.target,
                               args.gray, args.hsv, args.min_area, args.keep_ratio)
        if v == "SKIP":
            skipped.append((r.get("样本ID", "?"), detail))
            continue
        conf[(truth, v)] = conf.get((truth, v), 0) + 1
        done += 1

    if not done:
        print("[!] 没有任何一条样本被判出结果。先检查图片路径与 --bin 是否正确。")
        for sid, why in skipped:
            print(f"    {sid}: {why}")
        return 2

    labels = ["SUCCESS", "EMPTY", "LOST", "INVALID"]
    print("\n" + "=" * 74)
    print("混淆矩阵（行=人工真值，列=判别结果）")
    print("=" * 74)
    print(f"  {'真值':<10}" + "".join(f"{c:>10}" for c in labels) + f"{'合计':>8}")
    for t in labels:
        tot = sum(conf.get((t, c), 0) for c in labels)
        print(f"  {t:<10}" + "".join(f"{conf.get((t, c), 0):>10}" for c in labels) + f"{tot:>8}")

    # ── 分项指标
    n_succ = sum(conf.get(("SUCCESS", c), 0) for c in labels)
    n_empty = sum(conf.get(("EMPTY", c), 0) for c in labels)
    n_lost = sum(conf.get(("LOST", c), 0) for c in labels)
    n_inv = sum(conf.get(("INVALID", c), 0) for c in labels)

    tp_empty = conf.get(("EMPTY", "EMPTY"), 0)
    tp_lost = conf.get(("LOST", "LOST"), 0)
    tp_inv = conf.get(("INVALID", "INVALID"), 0)
    # 误报：本来成功，却被判成抓空/丢失/无帧
    fp_succ = n_succ - conf.get(("SUCCESS", "SUCCESS"), 0)

    print("\n" + "=" * 74)
    print("分项指标（每个指标单独报告，不合并成单一准确率）")
    print("=" * 74)
    lines = []
    lo, hi = wilson(tp_empty, n_empty)
    lines.append(("抓空检出率 TPR(EMPTY)", tp_empty, n_empty, lo, hi, GATE["tpr_empty"]))
    lo2, hi2 = wilson(fp_succ, n_succ)
    lines.append(("成功样本误报率 FAR", fp_succ, n_succ, lo2, hi2, GATE["far_success"]))
    lo3, hi3 = wilson(tp_lost, n_lost)
    lines.append(("推移/掉落检出率 TPR(LOST)", tp_lost, n_lost, lo3, hi3, GATE["tpr_lost"]))
    lo4, hi4 = wilson(tp_inv, n_inv)
    lines.append(("取帧异常检出率 TPR(INVALID)", tp_inv, n_inv, lo4, hi4, GATE["tpr_invalid"]))

    for name, k, n, lo_, hi_, thr in lines:
        print(f"  {name:<26} {pct(k, n):<16} 95%CI [{lo_:.2f}, {hi_:.2f}]"
              f"   门限 {thr:.2f}")

    # ── 门禁判定
    print("\n" + "=" * 74)
    print("门禁判定")
    print("=" * 74)
    checks = []
    checks.append((f"抓空检出率 ≥ {GATE['tpr_empty']:.2f}",
                   n_empty > 0 and tp_empty / n_empty >= GATE["tpr_empty"]))
    checks.append((f"抓空检出率 CI 下限 ≥ {GATE['tpr_empty_lo']:.2f}",
                   n_empty > 0 and wilson(tp_empty, n_empty)[0] >= GATE["tpr_empty_lo"]))
    far = fp_succ / n_succ if n_succ else 1.0
    checks.append((f"成功样本误报率 ≤ {GATE['far_success']:.2f}", n_succ > 0 and far <= GATE["far_success"]))
    checks.append((f"成功样本误报率 CI 上限 ≤ {GATE['far_success_hi']:.2f}",
                   n_succ > 0 and wilson(fp_succ, n_succ)[1] <= GATE["far_success_hi"]))
    checks.append((f"推移/掉落检出率 ≥ {GATE['tpr_lost']:.2f}",
                   n_lost > 0 and tp_lost / n_lost >= GATE["tpr_lost"]))
    checks.append((f"取帧异常检出率 ≥ {GATE['tpr_invalid']:.2f}",
                   n_inv > 0 and tp_inv / n_inv >= GATE["tpr_invalid"]))
    checks.append((f"SUCCESS 样本量 ≥ {GATE['n_success']}", n_succ >= GATE["n_success"]))
    checks.append((f"EMPTY 样本量 ≥ {GATE['n_empty']}", n_empty >= GATE["n_empty"]))
    checks.append((f"LOST 样本量 ≥ {GATE['n_lost']}", n_lost >= GATE["n_lost"]))
    checks.append((f"INVALID 样本量 ≥ {GATE['n_invalid']}", n_inv >= GATE["n_invalid"]))

    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")

    passed = all(ok for _, ok in checks)
    print("\n" + "=" * 74)
    if passed:
        print("结论：通过判别门禁，可进入真机联调。")
    else:
        print("结论：未通过判别门禁。按方案规则，未通过即不得进入正式实验——")
        print("      先补样本或调阈值重测；判定层不具备可信度时，")
        print("      重试会把误判放大成错误动作，其危害大于不判定。")
    print("=" * 74)
    print("注意：本报告只测判别器，不测策略。策略能否在只剩 1 个方块时改抓它，")
    print("      须按「上机前门禁」单独做 5 次试跑验收，两者不可互相替代。")

    # ── 落盘
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["指标", "分子", "分母", "比率", "CI下限", "CI上限", "门限", "是否达标"])
            for (name, k, n, lo_, hi_, thr), (cname, cok) in zip(lines, checks[:4]):
                w.writerow([name, k, n, f"{k / n:.4f}" if n else "",
                            f"{lo_:.4f}", f"{hi_:.4f}", thr, "达标" if cok else "未达标"])
            w.writerow([])
            w.writerow(["门禁项", "是否通过"])
            for name, ok in checks:
                w.writerow([name, "通过" if ok else "未通过"])
            w.writerow([])
            w.writerow(["总体结论", "通过" if passed else "未通过"])
            if skipped:
                w.writerow([])
                w.writerow(["跳过样本", "原因"])
                for sid, why in skipped:
                    w.writerow([sid, why])
        print(f"\n报告已写入：{args.out}")

    if skipped:
        print(f"\n跳过 {len(skipped)} 条：")
        for sid, why in skipped[:10]:
            print(f"  {sid}: {why}")

    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description="判别器正反样本评估（离线）")
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("template", help="生成标注表模板")
    t.add_argument("--out", default="labels.csv")
    t.add_argument("--force", action="store_true")

    p = sub.add_parser("pair", help="从录像截取一对执行前后帧并登记")
    p.add_argument("--video", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--outdir", default="pairs")
    p.add_argument("--labels", default="labels.csv")
    p.add_argument("--pre-frame", type=int, default=3)
    p.add_argument("--post-frame", type=int, default=-1, help="默认取倒数第 3 帧")
    p.add_argument("--truth", default=None)
    p.add_argument("--scene", default=None)

    e = sub.add_parser("eval", help="跑评估并给出门禁判定")
    e.add_argument("--labels", default="labels.csv")
    e.add_argument("--bin", type=lambda s: tuple(int(x) for x in s.split(",")), default=None)
    e.add_argument("--target", type=int, default=1)
    e.add_argument("--gray", type=lambda s: tuple(int(x) for x in s.split(",")),
                   default=(30, 170, 100))
    e.add_argument("--hsv", type=lambda s: [int(x) for x in s.split(",")],
                   default=(166, 120, 140, 12, 255, 255),
                   help="h_lo>h_hi 表示色相绕回（红色骑在 0/180 两侧）；"
                        "v_lo=140/s_hi=255 与 run_trial 一致，适配偏暗现场光")
    e.add_argument("--min-area", type=int, default=600)
    e.add_argument("--keep-ratio", type=float, default=0.40)
    e.add_argument("--out", default="label_eval_report.csv")

    args = ap.parse_args()
    if args.cmd == "template":
        return cmd_template(args)
    if args.cmd == "pair":
        return cmd_pair(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
