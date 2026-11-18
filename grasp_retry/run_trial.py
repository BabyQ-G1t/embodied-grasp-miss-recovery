#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trial.py —— 单次桌面整理试验执行器（唯一生产入口，A／B 两组共用同一份代码）

适用阶段（重要）：
    本脚本只在【模型训练完成之后】的正式实验阶段使用。项目全程分三个阶段：
      阶段一 采集示教：用 LeLab 的遥操作录制界面（主臂带从臂），本脚本完全不启动。
      阶段二 训练模型：用 LeLab 的训练页，本脚本也不启动。
      阶段三 正式实验：本脚本启动，跑 A／B 两组，产出 runs.csv、trials.csv。
    同目录下两份历史脚本（grasp_retry.py 的 auto 分支、lelab_auto.py）均已停用，
    启动时即打印拦截信息并退出，避免误用。

一、实验设计（2026-11-07 23:50 修订：人工监督对照）
    桌上只放 1 个方块；P1–P4 为训练内摆位，P5／P6 为训练分布外的泛化诊断位。
    每个位置 A／B 各 5 次，共 60 次有效试验；同一位置内按 A→B→A→B… 交错、A 先。
    一轮完整抓放固定 15 s（--duration 15，全程不变）。
    两组都运行同一套双证据判别并落盘，**唯一差别是由谁做「继续 / 停止」决策**：
      A 人工监督对照组：每轮后先由人独立判断是否继续（此阶段不揭示机判结论，
        避免锚定），机判结论在人做出决定后才揭晓，仅用于计算逐轮一致率；
        人决定停止即收工，决定继续则再跑一轮，最多 3 轮。
      B 判别器组：SUCCESS 即停、EMPTY 重试、连续抓空超过上限放弃，最多 3 轮。
    两组共享完全相同的强制安全停止条件：执行前／后取帧失败或画面不可用、
    执行异常、停机未确认、LOST（方块被推走／掉落／遮挡）、连续抓空 > 2 次、
    轮次预算用尽。安全条件优先于任何人的判断，人工不得覆盖。
    语音指令层（agent/）只替换触发方式，不进网络、不进自变量；
    A 组需人工值守，不能无人值守批量跑。

二、成功定义（双证据，缺一不可）
    只判「原位置变空」不足以证明方块进了收纳区——可能被推走、掉到桌外或被遮挡。
      证据一  原位置变空：原位框内方块色像素相对占比 < --keep-ratio（默认 0.40）
      证据二  收纳区增料：区内方块色像素较本轮执行前增加 ≥ 0.5 个单块当量
    两项都成立 → SUCCESS；原位仍有 → EMPTY（抓空）；原位变空但未增料
    → LOST（推走／掉落／遮挡），此轮不计入成功，试验按有效失败处理。
    收纳区必须事先标定（--bin 或 --calibrate-bin），否则拒绝开跑。

三、指标体系（命名与层级见 v10.xlsx「指标字典」sheet）
    M4 避免推理次数（唯一一级成效指标，**非劣效口径**）：
        逐次实测 j = 实际执行轮数 − 首次成功轮次，即「成功后仍启动的推理次数」。
        人工监督组的人看到成功也会停，故不预设 A/B 存在方向性差异；
        主报告值 = 两组 mean(j)、差值 Δ = mean(j_A) − mean(j_B) 及其 95% 置信区间，
        以非劣效界值 δ = 0.5 次判定「自动判别能否替代人工监督而不增加空跑」：
        CI 下界 > −δ 即非劣效成立。**不得预设方向、不得作优效结论或宣称 B 优于 A。**
        不使用 cap−k 反事实外推；A3（安全中止未执行轮次）不计入 M4。
    逐轮一致率（M4 的方法学前提，必须随主结论一起报告）：
        A 组每轮把「人判继续/停止」与「机判应继续/停止」比对，LOST 等安全强制轮次
        不参与；一致率低则 M4 的组间比较失去「同一决策标准」的前提。
    人工介入次数 / 人判耗时：描述量，量化被替代的人工监督成本。
    M1 首轮成功率：p 的代理，校验量，两组应无差异。
    M3 首次成功轮次：校验 + 描述；KM 仅描述，删失机制须标注。
    M2 最终入区率：描述量，不预设饱和，不得写成「判别提升抓取能力」。
    A1 实际执行轮数 / A2 单次耗时：辅助描述量。

    ⚠ 判别层不改权重、不改输入分布，因此**不改变单次抓取成功率 p**。
    本实验的因果主张只落在「自动判别能否替代人工监督」的非劣效结论，
    不宣称提升抓取能力，也不宣称比人更省推理。

四、共享安全停止条件与重试口径
    LeLab 的推理请求只含任务描述，不接受「去抓第几个方块」的目标参数。
    连续抓空上限 MAX_EMPTY_STREAK = 2 是**两组共用**的强制终止条件：
    连续抓空 3 次即 give_up，人工监督组的人也不能选择继续。
    该设计使第 3 轮在两组都存在结构性删失，KM 报告须标注。

五、安全与作废
    自动模式超时会调用 /stop-inference 并二次确认停机；停机失败即判现场不安全，
    立即中止本批并提示人工断电。取帧失败、执行超时、停机失败的试验标「作废」
    写入 trials.csv，作废试验不进任何指标汇总。

用法（半自动，最稳）：
    python run_trial.py --calibrate-bin
    python run_trial.py --group MIX --pos P1 --count 5 --manual
    python run_trial.py --group A --pos P1 --trial 1 --manual
    python run_trial.py --derive-a       # 分层输出 M2/M3(KM)/M4 汇总与生存曲线
用法（全自动，脚本自己调 LeLab）：
    python run_trial.py --group MIX --pos P1 --count 5 \
        --policy-ref <本地模型目录> --duration 15
    ★ 自动模式两路相机名必须字面上等于 F1/F2，与模型 config.json 对齐。
不用带硬件跑通逻辑：
    python run_trial.py --selftest       # 只验证「原位变空」这一半证据

操作顺序（手动模式）：
    第 0 步  --calibrate-bin 标定收纳区，存 bin_roi.json
    第 1 步  启动 LeLab，确认 localhost:8000；关掉浏览器里的相机实时预览
    第 2 步  一轮完整抓放固定 15 s（--duration 15，全程不变，不要按轮数累加）
    第 3 步  桌面摆 1 个方块、清空收纳区，双击 run_trial_local.bat
             分组选 MIX（A→B→A→B… 交替、A 先）；位置填单个 P 编号；连续次数 5
    第 4 步  脚本自动拍基准帧，找方块与原位框，核对收纳区标定
    第 5 步  照控制台提示：在 LeLab 点「执行」，机械臂回待机后按回车
    第 6 步  脚本拍帧做双证据判定：
             B 组：SUCCESS 即结束本次；EMPTY 提示 retry；LOST 提示现场复位
             A 组：每轮后由人独立判断「继续 / 停止」，机判结论事后揭晓；
                   两组共享连续抓空 > 2、LOST、取帧失败等强制安全停止条件
    ★ 每次试验之间人工复位：方块归位、收纳区清空。

产出：
    runs.csv    每轮一行，含双证据原始量与动作标记
    trials.csv  每次试验一行，含 M1–M4、A1/A3、人工介入次数、逐轮一致率、人判耗时

口径提醒（答辩会被问）：
    ACT 无语言输入，LeLab /start-inference 只吃固定 task 文本，语音只能当启动开关。
    泛化 P5／P6 是训练分布外，若 p≈0 两组一起归零，只作边界诊断、不进主结论。
    A 组人工判断必须现场逐轮记录，不得事后补填；未满预算的样本单列敏感性分析。
"""

import argparse
import csv
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

# 必须用 127.0.0.1，不要用 localhost：
# 系统代理（Clash / 公司代理 / WorkBuddy 透明代理）会把 localhost 也转发出去，
# /health 因此返回 502 Bad Gateway，看起来像「LeLab 没启动」，实际是代理拦的。
HOST = "http://127.0.0.1:8000"
CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"
HERE = pathlib.Path(__file__).resolve().parent
BIN_JSON = HERE / "bin_roi.json"

# 显式禁用代理：本机调用永远直连，不受 HTTP_PROXY 影响
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 判定状态常量
SUCCESS, EMPTY, LOST, NONE, ABORT = "SUCCESS", "EMPTY", "LOST", "NONE", "ABORT"

# ── 两组的行为差异，只有这张表 ──────────────────────────────────────
GROUPS = {
    #  group: (停止决策者, 判定是否驱动控制, 默认轮数上限, 默认连续抓空重试次数)
    #  A = 人工监督对照：每轮后由人独立判断「继续 / 停止」，机判结论事后揭晓；
    #  B = 判别器驱动：SUCCESS 即停、EMPTY 重试、连续抓空放弃。
    #  两组共用同一份代码、同一套双证据判别，以及下面完全相同的强制安全停止条件；
    #  唯一自变量 = 由谁做「继续 / 停止」决策。驱动方式同为 auto，不引入第二重差异。
    "A": dict(decider="human",   act_on_verdict=False, max_attempts=3, retries=0),
    "B": dict(decider="verdict", act_on_verdict=True,  max_attempts=3, retries=2),
}

# 共享安全停止条件：连续抓空上限（两组一致，人工不得覆盖）
# 连续抓空 > 2 即 give_up；LOST / 取帧失败 / 执行异常 / 停机未确认同样为强制终止。
MAX_EMPTY_STREAK = 2

# M4 非劣效界值：允许 B 组相对人工监督基线多出的空跑推理次数上限。
# 判定规则：Δ = mean(j_A) − mean(j_B) 的 95% CI 下界 > −δ 即非劣效成立。
NONINFERIORITY_MARGIN = 0.5

# 收纳区增料判定：增量达到半个单块当量即认为"有一块进去了"
# 取 0.5 而非 1.0，是因为方块落进收纳区后相互遮挡或贴边，
# 投影面积通常小于单独摆放时的面积，取 1.0 会系统性漏判。
BIN_GAIN_UNIT = 0.5

# 清空判定门限：单块场景下 target=1，当量达到 1−margin 即记成功。
# 余量用于容忍方块贴壁、被收纳区边沿切掉一部分导致的投影面积偏小。
# 单块只有「0 块 / 1 块」两态，不需要双块时代的 0.35 大余量，
# 故收紧到 0.25（门限 0.75）；若现场单块当量稳定偏低，可回调到 0.30。
BIN_CLEAR_MARGIN = 0.25


# ======================================================================
# 判定：双证据（原位置变空 + 收纳区增料）
# ======================================================================

def mask_of(img, gray=None, hsv_range=None, dark_v=None):
    """方块=255 的二值图。

    gray=(v_lo,v_hi,s_max) 找"低饱和的灰/黑"；dark_v 找"够暗的像素"（不管饱和度）；
    hsv_range 找彩色（旧 109 数据集的红方块场景用）。多路给全时取并集。

    为什么必须加 dark_v（2026-11-08 踩的坑）：
        灰块放在蓝板上时是 H113/S94/V120，走 gray 一路没问题；
        可它一旦被放进**黄色收纳板**，黄光反射+自身阴影让同一块变成 H10/S190/V45——
        饱和度 190 直接被 gray 的 `S<100` 判掉，色相又落在红色区间但 V45 不够红，
        结果"方块明明在黄板里，掩码上却一个像素都没有"，收纳区增料恒为 0，
        判成 LOST/EMPTY。而"够暗"这一个特征不受底色影响：块 V45、桌面 V200+、
        蓝板 V190、黄板 V255，门限 100 落在两个量级中间，怎么打光都不会误伤。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = []
    if gray is not None:
        v_lo, v_hi, s_max = gray
        masks.append((((hsv[:, :, 2] > v_lo) & (hsv[:, :, 2] < v_hi) &
                       (hsv[:, :, 1] < s_max))).astype(np.uint8) * 255)
    if dark_v:
        masks.append((hsv[:, :, 2] <= int(dark_v)).astype(np.uint8) * 255)
    if hsv_range is not None:
        h_lo, s_lo, v_lo, h_hi, s_hi, v_hi = hsv_range
        if h_lo > h_hi:
            # 色相绕回：h_lo..179 与 0..h_hi 两段并集。
            # 红色在 OpenCV 的 H 轴上骑在 0/180 两侧——同一个红块，
            # 实拍时读到 H≈3，录像里读到 H≈178（白平衡不同），
            # 只写单侧区间会在光照变化后整块漏检。
            m1 = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo], np.uint8),
                             np.array([179, s_hi, v_hi], np.uint8))
            m2 = cv2.inRange(hsv, np.array([0, s_lo, v_lo], np.uint8),
                             np.array([h_hi, s_hi, v_hi], np.uint8))
            masks.append(cv2.bitwise_or(m1, m2))
        else:
            lo = np.array(hsv_range[:3], np.uint8)
            hi = np.array(hsv_range[3:], np.uint8)
            masks.append(cv2.inRange(hsv, lo, hi))
    if not masks:
        m = np.zeros(hsv.shape[:2], np.uint8)
    else:
        # 三路（gray / dark / hsv）都要并进去，不能只 or 前两路
        m = masks[0]
        for extra in masks[1:]:
            m = cv2.bitwise_or(m, extra)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))


def find_blocks(m, min_area=600, max_area=12000, margin=12):
    """只认"实心、接近正方、不贴边"的连通域，滤掉桌沿阴影和长条。"""
    H, W = m.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        a = int(st[i, 4])
        if not (min_area <= a <= max_area):
            continue
        x, y, w, h = (int(v) for v in st[i, :4])
        if a / max(1, w * h) < 0.60:                 # 实心度
            continue
        if not (0.55 <= w / max(1, h) <= 1.8):       # 长宽比
            continue
        if x < margin or y < margin or x + w > W - margin or y + h > H - margin:
            continue
        out.append((x, y, w, h))
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def select_blocks(cands, target):
    """候选多于目标数时，挑面积最接近中位数的几个，避免把反光块当方块。"""
    if len(cands) <= target:
        return cands
    areas = sorted(w * h for _, _, w, h in cands)
    med = areas[len(areas) // 2] or 1
    return sorted(cands, key=lambda b: abs((b[2] * b[3]) / med - 1.0))[:target]


def ratio_in_box(m, box):
    """这个框里有多少像素"像方块"（0~1）。"""
    x, y, w, h = box
    patch = m[y:y + h, x:x + w]
    return float(patch.mean()) / 255.0 if patch.size else 0.0


def px_in_box(m, box):
    """这个框里"像方块"的像素个数，用于换算收纳区内的方块当量。"""
    x, y, w, h = box
    patch = m[y:y + h, x:x + w]
    return int(cv2.countNonZero(patch)) if patch.size else 0


def overlap_ratio(b, roi):
    """候选框落在收纳区内的面积占比，用于把收纳区里的方块从桌面候选里剔除。"""
    x, y, w, h = b
    X, Y, W, H = roi
    inter = max(0, min(x + w, X + W) - max(x, X)) * max(0, min(y + h, Y + H) - max(y, Y))
    return inter / max(1, w * h)


def pick_roi_of():
    """从 bin_roi.json 里读蓝板（放置区）ROI；老标定文件没有就返回 None。"""
    if not BIN_JSON.exists():
        return None
    try:
        p = json.loads(BIN_JSON.read_text(encoding="utf-8")).get("pick")
        return tuple(int(v) for v in p) if p and len(p) == 4 else None
    except Exception:
        return None


def play_area(bin_roi, pick_roi=None):
    """两块板的外接矩形 = 方块可能出现的地方。

    为什么要它（2026-11-08 的坑）：加了 dark_v 之后，机械臂黑夹爪（V≈20）也会被
    认成"深色方块"，在 t0 帧里多出一个 y≈17 的候选框，select_blocks 有可能把它
    当成本次的目标块——原位框画在夹爪上，后面怎么判都是错的。
    方块只会出现在蓝板或黄板上，用两块板的外接框一刀切掉画面上沿就够，
    比调暗门限更可靠（夹爪和方块都是黑的，靠亮度根本分不开）。
    """
    if pick_roi:
        x0 = min(bin_roi[0], pick_roi[0])
        y0 = min(bin_roi[1], pick_roi[1])
        x1 = max(bin_roi[0] + bin_roi[2], pick_roi[0] + pick_roi[2])
        y1 = max(bin_roi[1] + bin_roi[3], pick_roi[1] + pick_roi[3])
        return (x0, y0, x1 - x0, y1 - y0)
    return tuple(bin_roi)


def in_area(box, area):
    """框中心是否落在指定矩形内（用中心而不是重叠率，避免大框蹭边就通过）。"""
    x, y, w, h = box
    X, Y, W, H = area
    cx, cy = x + w / 2, y + h / 2
    return X <= cx <= X + W and Y <= cy <= Y + H


def target_boxes(m, bin_roi, pick_roi=None, target=1, min_area=600):
    """从掩码里挑出"桌面上还没进收纳区的目标方块"。返回 (全部候选, 选中的框)。

    三重过滤：① 方块形状（find_blocks）② 不能在收纳区里（已经在里面算入区，不算原位）
    ③ 中心必须在两块板范围内（踢掉上沿的机械臂/线缆）。
    """
    cands = find_blocks(m, min_area)
    area = play_area(bin_roi, pick_roi)
    table = [b for b in cands if overlap_ratio(b, bin_roi) < 0.5 and in_area(b, area)]
    if not table:
        # 方块被摆到两块板之外时不能直接判"没找到"，退回只做收纳区过滤的旧行为，
        # 由调用方按候选数报警，而不是静默失败。
        table = [b for b in cands if overlap_ratio(b, bin_roi) < 0.5]
    return cands, select_blocks(table, target)


def unit_px_of(m, boxes):
    """一个方块在画面上的单块当量（像素数），取各方块的中位数。"""
    vals = sorted(px_in_box(m, b) for b in boxes)
    return float(vals[len(vals) // 2]) if vals else 0.0


def bin_equiv(bin_px, unit_px):
    """收纳区内的方块当量（可带小数）。这是 M2 的唯一计量口径。"""
    return (bin_px / unit_px) if unit_px > 0 else 0.0


def cleared_of(eq, target):
    """清空判定：当量达到 target−margin 即记清空，避免叠放/贴壁导致漏判。"""
    return eq >= target - BIN_CLEAR_MARGIN


def check_boxes(before_mask, after_mask, boxes, keep_ratio=0.40):
    """
    证据一：逐个原始位置检查是否变空。
    keep_ratio=0.40 表示"现在的占比不到当初的 40% 就当已离开原位"。
    实测：灰方块在原位占比 79.3%，拿走后 16.0%（相对值 0.20），门限两侧各留一倍余量。
    注意：本函数只回答"是否离开原位"，不代表"是否已进入收纳区"。
    """
    status = []
    for b in boxes:
        r0 = ratio_in_box(before_mask, b)
        r1 = ratio_in_box(after_mask, b)
        rel = r1 / r0 if r0 > 1e-6 else 0.0
        status.append(dict(box=b, r0=r0, r1=r1, rel=rel, gone=rel < keep_ratio))
    return status


def judge(status, bin_gain, unit_px):
    """
    证据一 + 证据二 -> 本轮判定。
    bin_gain : 收纳区像素较本轮执行前的增量
    unit_px  : 单块当量像素数
    返回 SUCCESS / EMPTY / LOST。
    """
    gone = [s["gone"] for s in status]
    need = BIN_GAIN_UNIT * unit_px
    if not any(gone):
        return EMPTY                      # 原位还有方块，说明没抓到
    if bin_gain >= need:
        return SUCCESS                    # 离开原位且收纳区确实多了东西
    return LOST                           # 离开原位但收纳区没增料：推走／掉落／遮挡


# ======================================================================
# 取帧
# ======================================================================

def _ensure_schema(path, header):
    """表头与本版本不一致（旧版遗留文件）时先备份重建，返回 True 表示本次要写表头。

    2026-11-08 踩过的坑：trial_logs/trials.csv 表头是 17 列的旧版，新代码按 25 列
    追加，整行错位——「抓空次数」读到 A3（恒 0）、「A1 实际执行轮数」直接不存在，
    前端就显示成 0 和 ?。表头不一致必须换表，不能继续追加。
    """
    if not path.exists():
        return True
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            got = next(csv.reader(fh), [])
    except Exception:
        got = []
    if got == header:
        return False
    bak = path.with_name(f"{path.stem}_旧表头_{time.strftime('%Y%m%d_%H%M%S')}.bak.csv")
    try:
        path.replace(bak)
        print(f"[!] {path.name} 表头与当前版本不一致（旧 {len(got)} 列 / 现 {len(header)} 列），"
              f"已备份为 {bak.name}，本次重建表。")
    except Exception as e:
        print(f"[!] {path.name} 表头不一致且备份失败（{e}），继续追加会导致整行错位。")
    return True


def open_cam(index, width=640, height=480, warm=8, tries=10, gap=0.9):
    """打开相机取一帧，取完立刻释放。

    为什么要重试这么多次：
        LeLab 每次推理都是一个 lerobot_rollout 子进程，相机由该进程持有，
        进程 teardown 时 robot.disconnect() 才 release。从子进程退出到
        Windows 真正释放 DSHOW 句柄存在几秒延迟，若恰好在这个窗口取帧，
        第一次必然失败。因此这里按 tries × gap 约 9 秒的窗口反复尝试。
    """
    for i in range(tries):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            f = None
            for _ in range(warm):
                ok, tmp = cap.read()
                if ok:
                    f = tmp
                time.sleep(0.05)
            cap.release()
            if f is not None:
                if i:
                    print(f"    （相机在等待 {i * gap:.1f} 秒后可用，已取到画面）")
                return f
        else:
            cap.release()
        if i == 0:
            print(f"    相机 index {index} 暂不可用，等待释放（最长约 {tries * gap:.0f} 秒）…")
        time.sleep(gap)
    return None


def imwrite_u(path, img):
    """Windows 中文路径下 cv2.imwrite 会静默失败，必须走 imencode。"""
    ext = pathlib.Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        pathlib.Path(path).write_bytes(buf.tobytes())


def frame_quality(img):
    """
    取帧可用性检查。返回 (是否可用, 原因)。
    相机被占用、USB 掉线、自动曝光未收敛时，OpenCV 仍会返回一帧，
    但内容是全黑或近乎无纹理的噪声。若不拦掉，这类帧会被当成"方块不在原位"，
    直接酿成假成功。故在判定前先做可用性门禁。
    """
    if img is None:
        return False, "无画面"
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, std = float(g.mean()), float(g.std())
    if mean < 12:
        return False, f"画面过暗(均值{mean:.1f})"
    if mean > 245:
        return False, f"画面过曝(均值{mean:.1f})"
    if std < 4:
        return False, f"画面无纹理(标准差{std:.1f})"
    return True, "ok"


# ======================================================================
# 收纳区标定
# ======================================================================

def load_bin_roi(cli_bin):
    """收纳区优先级：命令行 > bin_roi.json。两者都没有则返回 None（脚本不许开跑）。"""
    if cli_bin:
        return tuple(cli_bin)
    if BIN_JSON.exists():
        try:
            d = json.loads(BIN_JSON.read_text(encoding="utf-8"))
            b = d.get("bin")
            if b and len(b) == 4:
                return tuple(int(v) for v in b)
        except Exception:
            pass
    return None


def save_bin_roi(roi):
    # 合并写入：bin_roi.json 里还有 pick 区与阈值口径，手标重跑标定时不能把它们冲掉
    d = {}
    if BIN_JSON.exists():
        try:
            d = json.loads(BIN_JSON.read_text(encoding="utf-8"))
        except Exception:
            d = {}
    d["bin"] = list(roi)
    d["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    BIN_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def calibrate_bin(args):
    """交互框选收纳区，保存为 bin_roi.json。收纳区移位后必须重跑本命令。"""
    f = open_cam(args.cam)
    if f is None:
        print(f"[!] 相机 {args.cam} 打不开，无法标定。先确认 LeLab 未占用相机。")
        return 2
    print("\n即将弹出取景窗口：用鼠标拖出一个矩形，框住【收纳区内部】。")
    print("要求：① 框内不含方块以外的杂物；② 不要把收纳区外沿框进来；")
    print("      ③ 框要略小于收纳区，宁可小一点，也不要框进桌面。然后按回车确认。\n")
    try:
        roi = cv2.selectROI("框选收纳区", f, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[!] 无法打开取景窗口（{e}）。请在有桌面环境的机器上运行，"
              f"或直接用 --bin x,y,w,h 手工给坐标。")
        return 2
    x, y, w, h = (int(v) for v in roi)
    if w < 20 or h < 20:
        print("[!] 框太小或未选中，标定取消。")
        return 2
    save_bin_roi((x, y, w, h))
    prev = f.copy()
    cv2.rectangle(prev, (x, y), (x + w, y + h), (0, 0, 255), 2)
    imwrite_u(HERE / "bin_roi_preview.png", prev)
    print(f"\n已保存收纳区 {x},{y},{w},{h} -> {BIN_JSON}")
    print(f"预览图：{HERE / 'bin_roi_preview.png'}（请打开确认红框正好套住收纳区）")
    print("\n下一步：摆 1 个方块到桌面、清空收纳区，再跑 run_trial.py 做真机联调。")
    return 0


# ======================================================================
# LeLab 接口
# ======================================================================

def api(path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode("utf-8", errors="replace")[:300]}
    except Exception as e:
        return {"_error": "conn", "_body": str(e)}


def still_running():
    """当前是否有推理在跑。读不到状态时按"仍在跑"处理，宁严勿松。

    字段坑（2026-11-08 定位）：LeLab 的 /inference-status 真实键名是
    `inference_active`（见 lelab/rollout.py handle_inference_status），
    早期代码读 active / running —— 两个键都不存在，于是恒判「已停」：
      · wait_idle 在 start 之后立刻返回 ok，不等 15 s 的 rollout 跑完；
      · 下一轮 start 撞 LeLab 的 409 "Inference is already active"；
      · 急停的二次确认变成假阳性（页面说已停，机械臂还在动）。
    兼容取第一个存在的键；全都认不出时返回 None（不确定），
    由调用方按"仍在跑"处理 —— 安全方向必须偏保守。
    """
    s = api("/inference-status", timeout=5)
    if s.get("_error"):
        return None
    for k in ("inference_active", "active", "running"):
        if k in s:
            return bool(s[k])
    return None


def wait_idle(timeout=300):
    """等待执行结束。返回 'ok' / 'timeout' / 'conn'，三者处置不同。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = still_running()
        if r is None:
            return "conn"
        if not r:
            return "ok"
        time.sleep(1.0)
    return "timeout"


def stop_arm(settle=45):
    """
    主动停机并二次确认。超时不是"停止等待"，而是必须让机械臂真的停下。
    返回 True 表示已确认停止；False 表示停机失败，现场不安全。
    """
    print("  [安全] 已超时，正在调用 /stop-inference 停止机械臂…")
    r = api("/stop-inference", {}, timeout=15)
    if r.get("_error"):
        print(f"  [安全] 停止接口调用失败：{r}")
    # 二次确认：接口返回成功不等于机械臂已停，必须看到状态转为空闲
    for _ in range(settle):
        s = still_running()
        if s is None:
            print("  [安全] 状态接口不可达，无法确认是否已停。")
            return False
        if not s:
            print("  [安全] 已确认停止。")
            return True
        time.sleep(1.0)
    print("  [安全] 停止后状态仍显示运行中。")
    return False


# ======================================================================
# 一次策略执行（两种驱动方式）
# ======================================================================

def run_once_manual(n_attempt, cap=None, left=None, in_bin=None, act=None):
    """手动模式的每轮提示。把"第几轮、还剩几个、这轮之后会怎样"一次说清。
    返回真正开始执行的时刻，用于把人工反应时间排除在耗时统计之外。
    """
    head = f"\n  >>> 第 {n_attempt}" + (f"/{cap}" if cap else "") + " 轮"
    if left is not None:
        head += f" ｜ 原位还剩 {left} 个方块"
    if in_bin is not None:
        head += f" ｜ 收纳区已进 {in_bin} 个"
    line = head + " ｜ 请在 LeLab 里点“执行”，等机械臂回到待机后按回车…"
    if act is False:
        line += ("\n      本组为人工监督对照：下一轮开始前会请你独立判断继续/停止，"
                 "机判结论届时揭晓；照常点执行即可。")
    input(line)
    return time.time()


def ensure_idle(timeout=15):
    """start 之前确认 LeLab 上没有残留任务。

    上一次试验被急停 / 崩溃打断时，LeLab 的 inference_active 可能仍是 True；
    此时直接 start 会吃 409，并白白把一次试验记成「作废」。
    顺序：先等 → 还在跑就停 → 再等。返回 True 表示确认空闲。
    """
    got = wait_idle(timeout=timeout)
    if got == "ok":
        return True
    if got == "conn":
        return False
    print(f"  [安全] LeLab 上仍有任务在跑（{timeout}s 内未自然结束），先停掉再开始")
    if not stop_arm():
        return False
    return wait_idle(timeout=20) == "ok"


def run_once_auto(req, duration):
    """自动执行一次。返回 (状态, 开始时刻)。

    状态 ∈ {'ok','timeout','conn','launch','busy','unsafe'}：
      busy   本轮根本没执行 —— LeLab 被残留推理 / 遥操作 / 录制占着
      launch start-inference 返回了 409 之外的真错误
    """
    if not ensure_idle():
        return "busy", None
    r = api("/start-inference", req, timeout=30)
    if r.get("_error"):
        # 409 有三个来源（见 lelab/rollout.py handle_start_inference）：
        # 推理未结束 / 遥操作占用 / 录制占用。只有第一种能靠 stop 清掉，
        # 所以只重试一次；仍失败就如实返回，不要把试验算成执行异常。
        if r.get("_error") == 409:
            print(f"  [安全] start 返回 409（{str(r.get('_body', ''))[:70]}），停一次再重试")
            if ensure_idle(timeout=5):
                r = api("/start-inference", req, timeout=30)
        if r.get("_error"):
            print(f"  [!] 启动失败：{r}")
            return ("busy" if r.get("_error") == 409 else "launch"), None
    t0 = time.time()
    st = wait_idle(timeout=duration + 180)
    if st == "timeout":
        if not stop_arm():
            return "unsafe", t0
    return st, t0


# ======================================================================
# 主流程
# ======================================================================

def ask_human(n, cap, detail, bin_gain, eq, frame_path, target):
    """A 组人工监督：先让人独立判断「继续 / 停止」，此阶段不揭示机判结论。

    机判结论由调用方在人做出决定后才揭晓，避免人被判别器锚定，
    逐轮一致率才有意义。返回 (continue_bool, 耗时秒, 理由)。
    """
    print()
    print("  " + "-" * 58)
    print(f"  第 {n}/{cap} 轮 · 人工监督决策（A 组）")
    print(f"  原始量：原位相对占比 {detail or 'n/a'}")
    print(f"          收纳区增量 {bin_gain:+d} px，当前当量 {eq:.2f}/{target} 块")
    print(f"  本轮关键帧：{frame_path}")
    print("  机判结论已封存，请先独立判断：c = 继续抓放 / s = 停止收工")
    t0 = time.time()
    try:
        raw = input("  你的决定 [c/s]（回车=c），可附理由：").strip()
    except EOFError:
        raw = "c"
    secs = time.time() - t0
    cont = not raw.lower().startswith("s")
    reason = raw[1:].strip(" :：") if len(raw) > 1 else ""
    print(f"  已记录：{'继续' if cont else '停止'}（决策耗时 {secs:.1f}s）")
    print("  " + "-" * 58)
    return cont, secs, reason


def run_trial(args):
    args.pos = args.pos or "P1"
    g = GROUPS[args.group]
    act = g["act_on_verdict"]
    decider = g["decider"]
    cap = args.max_attempts or g["max_attempts"]
    retries = MAX_EMPTY_STREAK      # 两组共享的强制安全上限；组参数/--retries 不再产生组间差异
    target = args.target

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"试验 {args.pos} / 第 {args.trial} 次    分组 = {args.group}")
    print(f"  停止决策者       : {'判别器（自动）' if decider == 'verdict' else '人（每轮独立判断，机判事后揭晓）'}")
    print(f"  轮数上限         : {cap} 轮")
    print(f"  共享强制安全上限 : 连续抓空 {retries} 次 / LOST / 取帧失败 / 停机未确认（两组一致，人工不得覆盖）")
    print(f"  成功定义         : 原位变空 且 收纳区增料 ≥ {BIN_GAIN_UNIT} 块当量（双证据）")
    print(f"  驱动方式         : {'手动（你点按钮）' if args.manual else '自动（调 LeLab 接口）'}")
    print("=" * 66)

    # 收纳区未标定不许开跑：没有收纳区就无法取得证据二，成功定义不成立
    if args.bin is None:
        print("[!] 尚未标定收纳区，脚本拒绝开跑。")
        print("    先执行：python run_trial.py --calibrate-bin")
        print("    或显式给：--bin x,y,w,h")
        return 2

    if not args.manual:
        print(f"  单次执行时长     : {args.duration} 秒（duration_s）")
        # 时长必须按"实测控制频率"标定，不能沿用旧的 15 秒：
        # 本机无 NVIDIA 显卡，LeLab 推理全程 CPU，实测 1.3–1.7 Hz；
        # 示范集 36 集 13863 帧 @30fps，平均 385 帧/集 → 需要约 385 个控制步，
        # 按 1.5 Hz 折算约 260 秒。2026-11-08 实跑：60 s 只走完前段（把方块推歪），
        # 280 s 完整抓起并放进收纳区。
        if args.duration < 240:
            print()
            print("  [!] duration 可能不够。本机 CPU 推理实测 1.3–1.7 Hz，")
            print("      示范轨迹约 385 步，需 ≈260 秒；时长不足只会走完轨迹前段。")
        elif args.duration > 400:
            print()
            print("  [!] duration 偏长。策略抓完后仍会继续输出动作，可能推乱方块。")
        h = api("/health", timeout=3)
        if h.get("_error"):
            print(f"[!] LeLab 服务没在跑：{h.get('_body', '')[:80]}")
            print("    先启动 LeLab；或用 --manual 走手动模式。")
            return 2
        if not args.policy_ref:
            print("[!] 自动模式必须给 --policy-ref。")
            return 2
        req = {
            "follower_port": args.port,
            "follower_config": args.config,
            "policy_ref": args.policy_ref,
            "task": args.task,
            "duration_s": args.duration,
            "cameras": args.cameras,
        }

    # —— 开局：拍基准帧，记下每个方块的位置与收纳区基线
    f0 = open_cam(args.cam)
    if f0 is None:
        print(f"[!] 相机 index {args.cam} 等了约 9 秒仍打不开。按顺序排查：")
        print("    ① 机械臂是否还在执行 —— 执行期间相机被 lerobot_rollout 子进程独占；")
        print("    ② 浏览器里 LeLab 页面是否停在有实时预览的界面 —— getUserMedia 会一直占着相机；")
        print("    ③ 换一个 --cam 编号试（0/1/2）；")
        print("    ④ 相机是否被其他软件（相机 App、录屏、会议软件）占用。")
        return 2

    okf, why = frame_quality(f0)
    if not okf:
        print(f"[!] 基准帧不可用（{why}）。相机可能被占用、USB 掉线或曝光未收敛。")
        print("    此时任何判定都不可信，故不写入任何试验记录。请处理后重跑。")
        return 2

    m0 = mask_of(f0, args.gray, args.hsv, args.dark_v)
    cands, boxes = target_boxes(m0, args.bin, pick_roi_of(), target, args.min_area)
    unit_px = unit_px_of(m0, boxes)
    bin_px0 = px_in_box(m0, args.bin)
    print(f"\n收纳区 {tuple(args.bin)}：开局方块色像素 {bin_px0}，单块当量 {unit_px:.0f} px")
    print(f"开局桌面候选 {len(cands)} 个，取 {len(boxes)} 个作为原位框：{boxes}")
    if unit_px <= 0:
        print("[!] 无法计算单块当量（原位框内没有方块色像素）。请先摆好方块，或调 --gray / --dark-v。")
        return 3
    if len(boxes) < target:
        print(f"[!] 预期 {target} 个，只可靠识别到 {len(boxes)} 个。")
        print("    请调整 --gray / --dark-v / --min-area，或先摆好方块再重跑。")
        return 3
    if bin_px0 > 0.35 * unit_px:
        print(f"[!] 收纳区里似乎还有东西（像素 {bin_px0}，约 {bin_px0 / unit_px:.1f} 块当量）。")
        print("    本实验以“进收纳区的块数”计成功，收纳区必须先清空；请清空后重跑。")
        return 3
    if len(cands) > target:
        print(f"    （候选多于目标数，已按面积接近度挑出 {target} 个，请核对坐标是否正好是方块）")
    imwrite_u(outdir / f"{args.pos}_{args.trial:02d}_t0.png", f0)

    alive = list(range(len(boxes)))       # 仍在原位的方块下标
    lost_idx = []                         # 离开原位但未进收纳区的方块
    results = []
    human_records = []           # A 组逐轮人判 / 机判比对
    empty_streak = 0
    void_reason = ""
    unsafe_stop = False                    # 停机未确认：整批中止，但仍须落盘留痕
    t_start = time.time()

    for n in range(1, cap + 1):
        # 本轮开始前，只对"还在原位的方块"重新量一次基准占比，并记收纳区基线
        f_pre = open_cam(args.cam)
        if f_pre is None:
            print("[!] 执行前取帧失败，本次试验作废，请人工处理后再继续。")
            void_reason = "执行前取帧失败"
            break
        okf, why = frame_quality(f_pre)
        if not okf:
            print(f"[!] 执行前画面不可用（{why}），本次试验作废。")
            void_reason = f"执行前画面不可用：{why}"
            break
        m_pre = mask_of(f_pre, args.gray, args.hsv, args.dark_v)
        boxes_alive = [boxes[i] for i in alive]
        bin_px_pre = px_in_box(m_pre, args.bin)

        if args.manual:
            t_exec = run_once_manual(n, cap, len(alive), bin_equiv(bin_px_pre, unit_px), act)
            st, t_exec = "ok", t_exec
        else:
            st, t_exec = run_once_auto(req, args.duration)
        if st == "unsafe":
            # 停机未确认：机械臂可能仍在运动，既不能再取帧也不能再调用策略。
            # 但“中止整批”不等于“不记录”——先补齐一条作废记录再退出，
            # 否则该次试验在 runs.csv／trials.csv 中根本不存在，
            # 门禁要求的“作废并记明原因”就无从追溯，事后无法分辨是漏做还是停机失败。
            results.append(dict(attempt=n, before=len(alive), after=len(alive),
                                verdict=ABORT, acted="unsafe", elapsed=0.0,
                                eq=bin_equiv(bin_px_pre, unit_px),
                                note="停机未确认，现场不安全"))
            void_reason = "停机后仍未确认停止（现场不安全，需断电核查）"
            unsafe_stop = True
            break
        if st == "busy":
            # 本轮根本没启动（LeLab 被残留推理 / 遥操作 / 录制占着）。
            # 第一轮就 busy = 整次试验尚未开始，直接按「前置条件不满足」退出，
            # 不写任何统计行；跑了几轮才 busy = 样本不完整，作废但留痕。
            print("  [!] LeLab 上有别的任务在跑（残留推理 / 遥操作 / 录制），本轮未执行。")
            print("      处理：LeLab 页面点 Stop，或重启 LeLab 服务后再来。")
            if n == 1 and not results:
                return 2
            results.append(dict(attempt=n, before=len(alive), after=len(alive),
                                verdict=ABORT, acted="abort", elapsed=0.0,
                                eq=bin_equiv(bin_px_pre, unit_px),
                                note="LeLab 被占用，本轮未执行"))
            void_reason = f"第 {n} 轮 LeLab 被别的任务占用，未执行"
            break
        if st != "ok":
            print(f"  [!] 执行未正常结束（{st}），本次试验作废，请人工处理。")
            results.append(dict(attempt=n, before=len(alive), after=len(alive),
                                verdict=ABORT, acted="abort", elapsed=0.0,
                                eq=bin_equiv(bin_px_pre, unit_px),
                                note=f"执行异常:{st}"))
            void_reason = f"执行异常:{st}"
            break

        if not boxes_alive:
            # 原位已无方块仍进入本轮：说明停止决策未及时做出，须如实计入 M4 的 j。
            results.append(dict(attempt=n, before=0, after=0, verdict=NONE,
                                acted="blind", elapsed=time.time() - (t_exec or time.time()),
                                eq=bin_equiv(bin_px_pre, unit_px), note="原位已空仍执行"))
            print(f"\n  第 {n} 轮 -> 原位已空，但停止决策未在此前做出，仍执行一次")
            continue

        time.sleep(1.0)
        f_post = open_cam(args.cam)
        if f_post is None:
            print("[!] 执行后取帧失败，本次试验作废，请人工处理后再继续。")
            void_reason = "执行后取帧失败"
            break
        okf, why = frame_quality(f_post)
        if not okf:
            print(f"[!] 执行后画面不可用（{why}），本次试验作废，判定不作数。")
            void_reason = f"执行后画面不可用：{why}"
            break
        m_post = mask_of(f_post, args.gray, args.hsv, args.dark_v)
        bin_px_post = px_in_box(m_post, args.bin)
        bin_gain = bin_px_post - bin_px_pre
        eq = bin_equiv(bin_px_post, unit_px)
        cleared_now = cleared_of(eq, target)

        status = check_boxes(m_pre, m_post, boxes_alive, args.keep_ratio)
        v = judge(status, bin_gain, unit_px)
        imwrite_u(outdir / f"{args.pos}_{args.trial:02d}_a{n}.png", f_post)

        detail = ", ".join(f"#{i}:{s['rel']:.2f}" for i, s in zip(alive, status))
        gone_idx = [alive[j] for j, s in enumerate(status) if s["gone"]]
        if v == SUCCESS:
            for i in gone_idx:
                alive.remove(i)
        elif v == LOST:
            # 离开原位却未进收纳区：不计入搬运，也不留在原位列表里，
            # 否则会被误当成"还在桌上待抓"，掩盖方块已丢失的事实。
            for i in gone_idx:
                alive.remove(i)
                lost_idx.append(i)

        if decider == "human":
            print(f"\n  第 {n} 轮 -> 机判结论已封存（人工监督组先独立判断）"
                  f"    原位相对占比 {detail}")
        else:
            print(f"\n  第 {n} 轮 -> 判定 {v}    原位相对占比 {detail}")
        print(f"     收纳区 {bin_px_pre} -> {bin_px_post} px（增量 {bin_gain:+d}，"
              f"当量 {eq:.2f} 块，增料门限 {BIN_GAIN_UNIT * unit_px:.0f} px）")
        if v == LOST:
            print("     ⚠ 方块离开了原位，但收纳区没有增料：可能被推走、掉到桌外，"
                  "或取帧时被遮挡。")
            print("     场景已不成立：本次试验立即结束，按「有效失败」记录，"
                  "不再继续调用策略。")
            print("     请先到现场确认方块去向并复位（方块归位、收纳区清空），"
                  "再开始下一次试验。")
        print(f"     原位剩余 {len(alive)} 个 ｜ 收纳区当量 {eq:.2f}/{target}"
              + (f" ｜ 丢失 {len(lost_idx)} 个" if lost_idx else ""))

        acted = "none"
        human_dec, human_secs, agree = "", 0.0, ""
        if v == LOST:
            # 场景本身已不成立，两组都必须立即结束；人工不得覆盖
            empty_streak = 0
            acted = "stop_lost"
        elif decider == "human":
            # 人工监督：先独立判断 → 再揭晓机判（避免锚定），只用于一致率
            hb, ht, hr = ask_human(
                n, cap, detail, bin_gain, eq,
                str(outdir / f"{args.pos}_{args.trial:02d}_a{n}.png"), target)
            machine_cont = (v != SUCCESS)
            agree = 1 if hb == machine_cont else 0
            human_dec = ("C" if hb else "S") + (f"({hr})" if hr else "")
            human_secs = ht
            human_records.append(dict(attempt=n, verdict=v, human_cont=hb,
                                      machine_cont=machine_cont, agree=agree,
                                      secs=ht, reason=hr))
            print(f"     机判结论揭晓：{v}（{'应继续' if machine_cont else '应停止'}）"
                  f"｜人判{'继续' if hb else '停止'}｜{'一致' if agree else '不一致'}")
            if v == EMPTY:
                empty_streak += 1
            else:
                empty_streak = 0
            # 共享强制安全上限：连续抓空 > 2 即 give_up，人工不得覆盖
            if empty_streak > retries:
                acted = "give_up"
            elif not hb:
                acted = "human_stop"
            elif n >= cap:
                acted = "exhausted"
            else:
                acted = "human_continue"
        elif act:
            if v == SUCCESS:
                empty_streak = 0
                acted = "continue"
            else:
                empty_streak += 1
                if empty_streak > retries:
                    acted = "give_up"                 # 连续抓空超过共享上限
                elif n < cap:
                    acted = "retry"
                else:
                    acted = "exhausted"               # 轮次预算用尽
        else:
            acted = "ignored"

        results.append(dict(attempt=n, before=len(boxes_alive), after=len(alive),
                            verdict=v, acted=acted,
                            human_dec=human_dec, human_secs=round(human_secs, 1),
                            agree=agree,
                            elapsed=max(0.0, time.time() - (t_exec or time.time())),
                            eq=eq,
                            note=";".join(f"{s['rel']:.2f}" for s in status)
                                 + f"|gain={bin_gain}"))

        # 清空判据：收纳区内的方块当量达到目标数，而不是"原位列表为空"。
        # 必须双证据成立（原位已空 + 收纳区增料）才允许提前结束：
        # 2026-11-08 实跑踩过——机械臂压过黄板时，黄板被压暗后命中灰掩码，
        # 收纳区瞬间 +5174 px（2.74 当量），但原位占比仍是 0.99，方块压根没动。
        # 旧逻辑只看单侧当量，于是打完 1 轮就宣布"已进入收纳区"，白扔 2 轮预算，
        # 而末尾复测 eq_end=0.00，汇总又写"清空失败"，前后自相矛盾。
        if cleared_now and not alive:
            if decider == "verdict" and act:
                print()
                print("  " + "=" * 58)
                print("   方块已进入收纳区，本次试验结束（判别器自动判定）")
                print("   不要再点“执行”了。")
                print(f"   共用了 {n} 轮/上限 {cap} 轮，省下 {cap - n} 轮")
                print("  " + "=" * 58)
                break
            if decider == "human":
                print(f"     ⓘ 收纳区已达目标，但停止决策由人做出"
                      f"（本轮人判：{'继续' if acted == 'human_continue' else '停止'}）；不自动终止")

        if acted == "stop_lost":
            print("     → 试验终止：场景不成立，剩余轮次不再执行。")
            break
        if acted == "give_up":
            print(f"     ⚠ 连续抓空 {empty_streak} 次，已达两组共享的强制上限 {retries} 次，"
                  f"放弃本试验（人工不得覆盖）。")
            break
        if acted == "human_stop":
            print("     → 人工监督判定停止，本次试验结束。")
            break
        if acted == "human_continue":
            print(f"     执行动作：人工监督决定继续（连续抓空 {empty_streak}/{retries}）")
        if acted == "exhausted":
            print(f"     ⓘ 轮次预算用尽，仍抓空，本次试验按清空失败计。")
        elif act and acted == "retry":
            print(f"     执行动作：retry（连续抓空 {empty_streak}/{retries}）")

    elapsed = time.time() - t_start
    # 最终以收纳区内的方块当量判定清空，与原位列表解耦
    if unsafe_stop:
        # 停机未确认时机械臂可能仍在运动，不再取帧，清空一律按未达标记
        eq_end = results[-1]["eq"] if results else 0.0
        cleared = 0
        okf, why = False, "停机未确认，未取收尾帧"
    else:
        f_end = open_cam(args.cam)
        okf, why = frame_quality(f_end)
        if okf:
            eq_end = bin_equiv(px_in_box(mask_of(f_end, args.gray, args.hsv, args.dark_v), args.bin), unit_px)
        else:
            eq_end = results[-1]["eq"] if results else 0.0
            if not void_reason:
                void_reason = f"收尾画面不可用（{why}），清空判定沿用末轮计数"
        cleared = 1 if cleared_of(eq_end, target) else 0
        # 轮级已判 SUCCESS 时，收尾帧不再二次设门槛：
        # SUCCESS 本身就要求「原位变空 + 收纳区增料 ≥ 0.5 当量」，而收尾帧走的是
        # cleared_of（≥ target−0.25 = 0.75）。2026-11-08 17:10 实跑踩到两套阈值打架：
        # 方块确实进了黄板，实测当量 0.72（灰块落在黄底上掩码偏小），
        # 于是轮级记 SUCCESS、试验级写"清空失败"，M2/M3 全错。
        if any(r["verdict"] == SUCCESS for r in results):
            cleared = 1
    n_human = len(human_records)
    agree_rate = (sum(h["agree"] for h in human_records) / n_human) if n_human else ""
    human_secs_total = sum(h["secs"] for h in human_records)
    human_seq = ",".join("C" if h["human_cont"] else "S" for h in human_records)
    n_empty = sum(1 for r in results if r["verdict"] == EMPTY)
    n_lost = sum(1 for r in results if r["verdict"] == LOST)
    # 因 LOST 提前终止的试验：仍为有效样本，但须在备注留痕，
    # 因为其执行轮数不代表策略用满了预算。
    lost_stop = any(r["acted"] == "stop_lost" for r in results)
    # M3 取"首次判定 SUCCESS 的轮次"为主口径（与轮级双证据一致），
    # 仅当没有任何轮次被评 SUCCESS 时才退回按当量比一次，避免再出现两套阈值。
    cleared_at = (next((r["attempt"] for r in results if r["verdict"] == SUCCESS), 0)
                  or next((r["attempt"] for r in results if cleared_of(r["eq"], target)), 0))
    # 显示用的整数块数：当量四舍五入后截到 target，仅用于表格阅读，
    # 是否清空一律以 cleared 字段为准，两者不混用。
    n_show = min(target, max(0, int(round(eq_end))))
    status_str = "有效" if not void_reason else "作废"

    print("\n" + "-" * 66)
    print(f"试验结束（{status_str}）：清空{'成功' if cleared else '失败'}，"
          f"共 {len(results)} 轮，抓空 {n_empty} 次，丢失 {n_lost} 次，耗时 {elapsed:.0f}s")
    if void_reason:
        print(f"  作废原因：{void_reason}")
    print("-" * 66)

    # —— 落盘
    run_csv = outdir / "runs.csv"
    RUN_HEAD = ["时间", "对局", "分组", "位置", "试验", "轮次",
                "原位方块", "收纳区当量", "判定", "采取动作",
                "人工决策", "人判耗时s", "人机一致1/0", "REL明细", "试验状态"]
    new_runs = _ensure_schema(run_csv, RUN_HEAD)
    with open(run_csv, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if new_runs:
            w.writerow(RUN_HEAD)
        ts = time.strftime("%H:%M:%S")
        for r in results:
            w.writerow([ts, f"{args.pos}-{args.trial:02d}", args.group, args.pos,
                        args.trial, r["attempt"], r["before"], f"{r['eq']:.2f}",
                        r["verdict"], r["acted"], r.get("human_dec", ""),
                        r.get("human_secs", ""), r.get("agree", ""),
                        r["note"], status_str])

    trial_csv = outdir / "trials.csv"
    TRIAL_HEAD = ["时间", "分组", "位置", "试验", "试验状态", "作废原因",
                  "M2 最终入区1/0", "A1 实际执行轮数", "M3 首次成功轮次",
                  "M4 成功后仍启动推理次数j", "A3 安全中止未执行轮次", "抓空次数",
                  "丢失次数", "M1 首轮成功1/0", "最终入区0-1", "收纳区当量",
                  "A2 耗时s", "名义预算cap", "驱动方式", "预算未满待核查",
                  "人工介入次数", "逐轮一致率", "人判耗时s", "人判决策序列", "备注"]
    new_t = _ensure_schema(trial_csv, TRIAL_HEAD)
    first_ok = 1 if (results and results[0]["verdict"] == SUCCESS) else 0
    n_exec = len(results)
    # M4 的实测观测量 j：首次成功之后仍然启动的推理次数（不是 cap−k 反事实外推）。
    # 两组都可能在首次成功后停止（A 由人判、B 由判别器），故不预设方向；
    # 主报告为两组 mean(j) 与 Δ 的置信区间，按非劣效口径判定。
    j_m4 = max(0, n_exec - cleared_at) if cleared_at else 0
    idle = j_m4                      # 保留旧变量名，供其它引用处复用
    # A3：安全中止（LOST / give_up / unsafe / abort）导致的未执行轮次，不计入 M4
    early_reason = any(r["acted"] in ("stop_lost", "give_up", "unsafe", "abort")
                       for r in results)
    a3 = max(0, cap - n_exec) if early_reason else 0
    drive = "手动" if args.manual else "自动"
    # 预算未满：非正常终止、非成功即停、非安全中止，须与录屏核对后单列敏感性分析
    under = "待核查" if (n_exec < cap and not early_reason and not cleared_at
                        and not void_reason) else ""
    with open(trial_csv, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if new_t:
            w.writerow(TRIAL_HEAD)
        w.writerow([time.strftime("%H:%M:%S"), args.group, args.pos, args.trial,
                    status_str, void_reason, cleared, n_exec, cleared_at, j_m4, a3,
                    n_empty, n_lost, first_ok, n_show, f"{eq_end:.2f}", f"{elapsed:.0f}",
                    cap, drive, under, n_human,
                    (f"{agree_rate:.2f}" if agree_rate != "" else ""),
                    f"{human_secs_total:.1f}", human_seq,
                    ("无有效轮次" if not results
                     else ("LOST 中止，未跑满预算（场景不成立，按有效失败计）"
                           if lost_stop else ""))])

    print(f"\n  M1 首轮成功 : {first_ok}（1/0）"
          f"   M2 最终入区 : {n_show}/{target}（当量 {eq_end:.2f}）"
          f"   M3 首次成功轮次 : {cleared_at or '未成功（右删失）'}"
          f"   M4 成功后仍启动推理 j : {j_m4}   A3 : {a3}   A2 耗时 {elapsed:.0f}s")
    print(f"  试验状态 : {status_str}" + (f"（{void_reason}）" if void_reason else ""))

    print(f"\n已写入：{run_csv}\n        {trial_csv}")
    if unsafe_stop:
        print("\n" + "=" * 66)
        print("[安全] 停机未确认，本次试验已记为「作废」并落盘。")
        print("       现场请立即断电或按下物理急停，确认机械臂完全静止、")
        print("       排除故障并由人工重新上电归零后，再开始下一次试验。")
        print("       本批其余试验全部中止，不得继续。")
        print("=" * 66)
        return 4
    return 0 if not void_reason else 5


# ======================================================================
# 无硬件自测：用本机已有录像验证"原位置变空"这一半证据
# ======================================================================

def selftest(args):
    root = CACHE / "shuren2"
    if not root.exists():
        print(f"[!] 没找到数据集目录 {root}")
        return 1
    print("用本机真实录像自测：首帧 vs 末帧，只验证「离开原位」这一半证据。\n"
          "注意：本自测只能说明原位判定可用，不能验证「进入收纳区」，\n"
          "      也不能给出抓空检出率与误报率——后者须用 label_eval.py 在正反样本上评估。\n")
    ok = bad = skip = 0
    for ds in sorted(root.iterdir()):
        vids = list((ds / "videos").rglob("*.mp4")) if (ds / "videos").exists() else []
        if len(vids) < 2:
            continue
        pick = None
        for v in vids:
            try:
                if "\u73af\u5883" in str(v.parent.parent.name):
                    pick = v
                    break
            except Exception:
                pass
        pick = pick or vids[0]
        cap = cv2.VideoCapture(str(pick))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n < 10:
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, 3)
        ok0, f_first = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 3))
        ok1, f_last = cap.read()
        cap.release()
        if not (ok0 and ok1):
            continue
        m_first = mask_of(f_first, args.gray, args.hsv, args.dark_v)
        m_last = mask_of(f_last, args.gray, args.hsv, args.dark_v)
        cands = find_blocks(m_first, args.min_area)
        boxes = select_blocks(cands, args.target)
        if not boxes:
            print(f"  {ds.name}: 首帧找不到方块 —— 该数据集与这组参数不匹配，跳过")
            skip += 1
            continue
        st = check_boxes(m_first, m_last, boxes, args.keep_ratio)
        good = any(s["gone"] for s in st)
        ok += good
        bad += not good
        rels = ", ".join(f"{s['rel']:.2f}" for s in st)
        print(f"  {ds.name}: {len(boxes)} 个方块 -> 离开原位 "
              f"{'是' if good else '否':<2}  REL=[{rels}]  {'✓' if good else '✗'}")
    print(f"\n合计：符合预期 {ok} 条，不符合 {bad} 条，参数不匹配跳过 {skip} 条")
    if bad:
        print("有不符合的先别上真机——把 --gray 或 --keep-ratio 调一遍再自测。")
    elif skip:
        print("跳过的那些不影响结论：正式实验只需要你自己的方块和机位能被稳定识别。")
    print("\n下一步（必须做，否则不算通过判别门禁）：")
    print("  1) 按「示教采集清单」补录抓空、遮挡、推移、读帧异常的负样本；")
    print("  2) 用 label_eval.py 在正反样本上算出检出率、漏检率、误报率。")
    return 0 if bad == 0 else 1


def km_curve(records, cap):
    """
    Kaplan-Meier 生存曲线（此处"生存"= 仍未清空）。
    records: [(首次成功轮次 或 cap, 是否事件)]，未成功记 (cap, False) 即右删失。
    返回 [(轮次 t, 风险集, 事件数 d, 删失数 c, 生存率 S(t))]，长度恒为 cap，
    即使窗口内已无风险样本也保留该行（S 不变），便于画完整阶梯图。
    """
    at_risk = len(records)
    surv = 1.0
    out = []
    for t in range(1, cap + 1):
        d = sum(1 for x, e in records if x == t and e)
        c = sum(1 for x, e in records if x == t and not e)
        if at_risk > 0 and d:
            surv *= (1.0 - d / at_risk)
        out.append((t, at_risk, d, c, surv))
        at_risk -= (d + c)
    return out


def km_median(records, cap):
    """
    Kaplan-Meier 中位生存时间：生存率首次跌破 0.5 的轮次。
    若曲线始终高于 0.5（多数样本到窗口末仍未清空）则返回 None。
    """
    for t, _risk, _d, _c, surv in km_curve(records, cap):
        if surv <= 0.5:
            return t
    return None


def median(vals):
    if not vals:
        return None
    v = sorted(vals)
    k = len(v)
    return v[k // 2] if k % 2 else (v[k // 2 - 1] + v[k // 2]) / 2


TRAIN_POS = ("P1", "P2", "P3", "P4")


def split_pos(pos):
    """把 'P1+P3' / 'P1,P3' / 'P1 P3' 这类多位置写法拆成位置列表。"""
    s = str(pos or "").strip().upper()
    for ch in "+/、,;|":
        s = s.replace(ch, " ")
    return s.split()


def stage_of(pos, train_pos=TRAIN_POS):
    """按摆放位置划分训练内／泛化。P1–P4 录过示教，P5、P6 未录，专用于泛化。

    单块实验只填 1 个位置（如 P1）；多值写法仅为兼容历史数据：
      全部落在示教位置   → 训练内
      全部落在示教之外   → 泛化
      一边一个           → 混合
    混合层单列、不进 KM 分层：一次试验只能归属一个覆盖条件，混算会把
    训练内与泛化两条生存曲线搅在一起，泛化差距就解释不清了。
    """
    ps = split_pos(pos)
    if not ps:
        return "泛化"
    train = set(train_pos)
    hit = [p in train for p in ps]
    if all(hit):
        return "训练内"
    if not any(hit):
        return "泛化"
    return "混合"


# ======================================================================
# 在线判别复核：混淆矩阵的真值载体
# ======================================================================
# 记录表一行只对应一次试验，而判别是逐轮发生的，一格填不了多轮。
# 故逐轮真值另存 review.csv：由 runs.csv 自动铺出待复核清单（试验、轮次、脚本判定、
# 前后帧文件名），现场只需看图填「人工真值」一列。
# A 与 B 的判别都照常运行并落盘，故两组的轮次一律进清单，混淆矩阵由两组共同构成。

REVIEW_HEAD = ["对局", "分组", "位置", "分层", "试验", "轮次", "脚本判定",
               "执行前帧", "执行后帧", "人工真值", "复核备注"]
TRUTH_OF = {SUCCESS: "成功", EMPTY: "抓空", LOST: "丢失"}
TRUTH_CHOICES = ("成功", "抓空", "丢失", "无帧", "不适用")


def export_review(outdir="trial_logs", train_pos=TRAIN_POS):
    """把 runs.csv 的逐轮记录铺成待复核清单 review.csv。

    只铺「有效」试验的轮次；作废试验整条剔除，与汇总口径一致。
    判定为 NONE 的轮次（原位已空仍盲目执行）没有判定对象，真值预填「不适用」，
    统计时自动排除，避免把它算成一次漏检。
    """
    src = pathlib.Path(outdir) / "runs.csv"
    if not src.exists():
        print(f"[!] 找不到 {src}，先跑几组试验再来。")
        return 2
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    if not rows:
        print("[!] runs.csv 是空的。")
        return 2

    # 已有 review.csv 时先读回来：批量试验要分几天跑，每次都重新导出，
    # 若直接覆盖，前面几天已填的人工真值会被抹掉，且没有任何报错提示。
    dst = pathlib.Path(outdir) / "review.csv"
    filled = {}
    if dst.exists():
        for old in csv.DictReader(open(dst, encoding="utf-8-sig")):
            key = (old.get("对局", ""), old.get("轮次", ""))
            if (old.get("人工真值") or "").strip() or (old.get("复核备注") or "").strip():
                filled[key] = ((old.get("人工真值") or "").strip(),
                               old.get("复核备注") or "")

    out_rows, missing, kept = [], 0, 0
    for r in rows:
        try:
            if r.get("试验状态", "有效") == "作废":
                continue
            pos = r["位置"]
            trial = int(r["试验"])
            n = int(r["轮次"])
            verdict = r["判定"]
        except (KeyError, ValueError):
            continue
        stem = f"{pos}_{trial:02d}"
        pre_f = f"{stem}_t0.png" if n == 1 else f"{stem}_a{n - 1}.png"
        post_f = f"{stem}_a{n}.png"
        for f in (pre_f, post_f):
            if not (pathlib.Path(outdir) / f).exists():
                missing += 1
        prefill = "不适用" if verdict == NONE else ""
        truth, memo = filled.get((r.get("对局", ""), str(n)), (prefill, ""))
        if truth:
            kept += 1
        out_rows.append([r.get("对局", ""), r["分组"], pos, stage_of(pos, train_pos),
                         trial, n, verdict, pre_f, post_f, truth, memo])

    with open(dst, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(REVIEW_HEAD)
        w.writerows(out_rows)
    todo = sum(1 for r in out_rows if not r[9])
    print(f"待复核清单已导出：{dst}")
    print(f"  共 {len(out_rows)} 个轮次，其中 {todo} 个待填人工真值"
          f"（{len(out_rows) - todo} 个已填或为「不适用」）")
    if kept:
        print(f"  已从上一版 review.csv 保留 {kept} 条已填真值，未被覆盖")
    print(f"  人工真值可选：{' / '.join(TRUTH_CHOICES)}；"
          f"判定口径见「逐轮判别复核」页说明")
    if missing:
        print(f"  ⚠ 有 {missing} 个帧文件在 {outdir} 下找不到，复核前请确认是否被清理或未保存")
    print("  填完真值后运行：python run_trial.py --confusion --review-dir "
          f"{outdir}")
    return 0


def confusion(outdir="trial_logs"):
    """按 review.csv 的人工真值统计在线判别混淆矩阵，分项报告，不合并为单一准确率。

    分母口径（与方案指标第 6 条一致）：
      抓空检出率 = P(判抓空 | 真值抓空)；漏检率 = P(判成功 | 真值抓空)；
      成功确认率 = P(判成功 | 真值成功)；误报率 = P(判抓空或丢失 | 真值成功)；
      丢失检出率 = P(判丢失 | 真值丢失)。
    「无帧」与「不适用」两类不参与上述比例，单独计数。
    """
    src = pathlib.Path(outdir) / "review.csv"
    if not src.exists():
        print(f"[!] 找不到 {src}，先运行 --export-review 铺出待复核清单。")
        return 2
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    if not rows:
        print("[!] review.csv 是空的。")
        return 2

    table = {}          # (分层, 分组) -> {(真值, 脚本判定): 计数}
    pending = {}
    n_na = 0
    for r in rows:
        truth = (r.get("人工真值") or "").strip()
        key = ((r.get("分层") or "").strip(), (r.get("分组") or "").strip())
        if truth == "不适用":
            n_na += 1                 # 无判定对象的轮次，既不算对也不算错，单列计数
            continue
        if truth not in TRUTH_CHOICES:
            pending[key] = pending.get(key, 0) + 1      # 尚未复核
            continue
        cell = table.setdefault(key, {})
        cell[(truth, (r.get("脚本判定") or "").strip())] = \
            cell.get((truth, (r.get("脚本判定") or "").strip()), 0) + 1

    if not table:
        print("[!] 还没有任何已填真值的轮次，无法统计。")
        return 2

    verdicts = [SUCCESS, EMPTY, LOST]
    print("=" * 78)
    print("在线判别混淆矩阵（真值取人工复核，脚本判定取 runs.csv；两组共同构成）")
    print("=" * 78)
    print(f"  {'分层·分组':<16}{'真值':<8}" + "".join(f"{v:<10}" for v in verdicts) + "合计")
    for key in sorted(table):
        cell = table[key]
        for truth in ("成功", "抓空", "丢失", "无帧"):
            sub = [cell.get((truth, v), 0) for v in verdicts]
            if not any(sub):
                continue
            print(f"  {'·'.join(key):<16}{truth:<8}"
                  + "".join(f"{x:<10}" for x in sub) + str(sum(sub)))

    print("-" * 78)
    tot = {}
    for cell in table.values():
        for k, v in cell.items():
            tot[k] = tot.get(k, 0) + v
    def rate(num_keys, den_truth, label):
        den = sum(v for (t, _v), v in tot.items() if t == den_truth)
        num = sum(v for (t, v_), v in tot.items() if t == den_truth and v_ in num_keys)
        if not den:
            print(f"  {label:<12} 无样本")
            return
        print(f"  {label:<12} {num}/{den} = {num / den:.1%}")
    rate([EMPTY], "抓空", "抓空检出率")
    rate([SUCCESS], "抓空", "漏检率")
    rate([SUCCESS], "成功", "成功确认率")
    rate([EMPTY, LOST], "成功", "误报率")
    rate([LOST], "丢失", "丢失检出率")

    n_ok = sum(v for (t, v_), v in tot.items() if TRUTH_OF.get(v_) == t)
    n_all = sum(tot.values())
    print(f"  {'总一致率':<12} {n_ok}/{n_all} = {n_ok / n_all:.1%}"
          f"（仅供参考，不作门禁：真值分布不均时会被大类主导）")
    if n_na:
        print(f"  另有 {n_na} 个轮次判为「不适用」（原位已空、无判定对象），不计入上述任何比例。")
    if pending:
        left = sum(pending.values())
        print(f"\n  ⚠ 还有 {left} 个轮次未复核（"
              + "、".join(f"{'·'.join(k)} {v}" for k, v in sorted(pending.items()))
              + "），未纳入上表。复核未完成时须注明矩阵只覆盖已完成部分，"
                "不得把未复核样本按一致处理。")
    print("\n  口径限制：本矩阵检验的是「判别器在真实推理流程与真实机位下的表现」，"
          "与 label_eval.py 的离线标定不是同一份证据；\n"
          "  离线门禁看的是阈值本身的判别力，在线矩阵看的是端到端链路的实际命中情况，两者须分别报告。")
    print("=" * 78)
    return 0


def _m4_report(outdir, train_pos):
    """M4 非劣效主报告：实测 j 的分组均值、Δ 的 95% CI 与非劣效判定。

    人工监督对照下 A 组的人也会在成功后停止，故不预设组间方向；
    结论只落在「自动判别能否替代人工监督而不增加空跑」的非劣效判定上。
    """
    import math
    p = pathlib.Path(outdir) / "trials.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    key = "M4 成功后仍启动推理次数j"
    buckets, aux = {}, {}
    for r in rows:
        if r.get("试验状态") != "有效" or key not in r:
            continue
        try:
            j = float(r[key])
        except (TypeError, ValueError):
            continue
        k = (stage_of(r.get("位置", ""), train_pos), r.get("分组", "?"))
        buckets.setdefault(k, []).append(j)
        a = aux.setdefault(k, dict(human=0, agree_n=0, agree_hit=0.0, secs=0.0))
        try:
            nh = int(float(r.get("人工介入次数") or 0))
        except ValueError:
            nh = 0
        a["human"] += nh
        try:
            v = r.get("逐轮一致率", "")
            if v not in ("", None) and nh:
                a["agree_n"] += nh
                a["agree_hit"] += float(v) * nh
        except ValueError:
            pass
        try:
            a["secs"] += float(r.get("人判耗时s") or 0)
        except ValueError:
            pass
    if not buckets:
        return

    def stats(v):
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        return n, m, sd

    fmt = lambda x: "n/a" if x != x else f"{x:.3f}"
    print("-" * 78)
    print("M4 避免推理次数（唯一一级成效指标 · 非劣效口径）")
    print(f"  非劣效界值 δ = {NONINFERIORITY_MARGIN} 次；判定：Δ 的 95% CI 下界 > −δ 即成立")
    print("  两组都可能在首次成功后停止（A 由人判、B 由判别器），不预设方向，不作优效结论")
    for st in ("训练内", "泛化"):
        a, b = buckets.get((st, "A"), []), buckets.get((st, "B"), [])
        if not a and not b:
            continue
        na, ma, sa = stats(a) if a else (0, float("nan"), 0.0)
        nb, mb, sb = stats(b) if b else (0, float("nan"), 0.0)
        delta = ma - mb if a and b else float("nan")
        ok = a and b and na > 1 and nb > 1
        se = math.sqrt(sa * sa / na + sb * sb / nb) if ok else float("nan")
        lo = delta - 1.96 * se if ok else float("nan")
        hi = delta + 1.96 * se if ok else float("nan")
        verdict = ("非劣效成立" if ok and lo > -NONINFERIORITY_MARGIN else
                   "未达非劣效" if ok else "样本不足／方差为 0，无法判定")
        print(f"  {st}：A n={na:>2} mean(j)={fmt(ma)} sd={fmt(sa)}   ｜   "
              f"B n={nb:>2} mean(j)={fmt(mb)} sd={fmt(sb)}")
        print(f"        Δ = A−B = {fmt(delta)}   95% CI [{fmt(lo)}, {fmt(hi)}]   → {verdict}")
        x = aux.get((st, "A"))
        if x and x["agree_n"]:
            print(f"        A 组方法学前提：人工介入 {x['human']} 次，"
                  f"逐轮一致率 {x['agree_hit'] / x['agree_n']:.2f}，"
                  f"人判耗时 {x['secs']:.1f}s")
    print("  口径：Welch 标准误 + 正态近似 1.96（n<30 时正式报告应改用 t 分位数）；")
    print("        j 逐次实测，不用 cap−k 外推；A3 不计入 M4；")
    print("        安全强制轮次（LOST / give_up）不参与逐轮一致率；")
    print("        未满名义预算且无合法中止的样本已标「待核查」，须单列敏感性分析。")


def derive_a(outdir="trial_logs", target=1, cap=3, train_pos=TRAIN_POS):
    """按 训练内／泛化 分层，输出 A 与 B 的 M2、M3（Kaplan-Meier）、生存曲线与 M4。

    单块 / 3 轮口径。M3 只是校验 + 描述量：两组首次成功轮次理论同分布，
    但 B 组连续抓空 2 次会 give_up，第 3 轮存在结构性（信息性）删失；
    两组共用连续抓空上限，第 3 轮均存在结构性删失。故 KM 曲线仅作描述，
    报告 M3 时必须标注删失机制，不得把组间中位数差当编排效果。

    删失口径：
      ① 窗口内成功 → 事件发生，时间取首次成功轮次；
      ② 窗口内未成功 → 右删失，时间取**该次试验实际执行到的末轮**，不按窗口上限外推；
      ③ 作废试验不进入任何汇总。
    训练内（P1–P4）与泛化（P5／P6）分别统计；泛化层只作边界诊断，不进主结论。
    """
    p = pathlib.Path(outdir) / "runs.csv"
    if not p.exists():
        print(f"[!] 找不到 {p}，先跑几组 A 与 B 再来。")
        return 2
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    if not rows:
        print("[!] runs.csv 是空的。")
        return 2

    trials = []
    prev_key, cur = None, None
    for r in rows:
        try:
            if r.get("试验状态", "有效") == "作废":
                continue                                    # 作废试验不进入任何汇总
            g = r["分组"]
            key = (g, r["位置"], int(r["试验"]))
            n = int(r["轮次"])
            bn = float(r["收纳区当量"])
        except (KeyError, ValueError):
            continue
        if key != prev_key or n == 1:
            cur = []
            trials.append((g, stage_of(r["位置"], train_pos), cur))
            prev_key = key
        cur.append((n, bn))
    for _g, _s, v in trials:
        v.sort()

    early_stop = 0            # 未清空且未跑满窗口的样本（give_up / LOST），删失时间须按实跑轮次
    for _g, _s, v in trials:
        if not v:
            continue
        if v[-1][0] < cap and not any(cleared_of(b, target) for _n, b in v):
            early_stop += 1

    def summarize(window, only_groups, stage, label):
        rec = dict(n=0, cleared=0, km=[], at=[], censored=0, early=0)
        for g, st, rows_ in trials:
            if g not in only_groups or (stage and st != stage):
                continue
            sub = [(n, b) for n, b in rows_ if n <= window]
            if not sub:
                continue
            hit = next((n for n, b in sub if cleared_of(b, target)), 0)
            rec["n"] += 1
            if hit:
                rec["cleared"] += 1
                rec["at"].append(hit)
                rec["km"].append((hit, True))
            else:
                # 删失时间 = 窗口内实际观察到的末轮，而不是窗口上限
                last = max(n for n, _b in sub)
                rec["censored"] += 1
                if last < window:
                    rec["early"] += 1
                rec["km"].append((last, False))
        return label, rec, window

    summaries = [
        summarize(cap, ("A",), "训练内", "训练内·A"),
        summarize(cap, ("B",), "训练内", "训练内·B"),
        summarize(cap, ("A",), "泛化", "泛化·A"),
        summarize(cap, ("B",), "泛化", "泛化·B"),
    ]

    mixed = [t for t in trials if t[1] == "混合"]
    if mixed:
        print(f"[!] 有 {len(mixed)} 次试验的方块跨层摆放（如 P1+P5），已单列为「混合」，"
              f"未计入下面任何一层。")
        print("    泛化结论只认整场都在 P5／P6 的试验；这批要么按同层重做，要么另立一层单独报告。")
        print()

    print("=" * 78)
    print("M3 分层汇总（Kaplan-Meier，按位置分训练内／泛化；作废试验已剔除）")
    print("=" * 78)
    for label, rec, window in summaries:
        if not rec["n"]:
            print(f"  {label:<26} 无数据")
            continue
        km = km_median(rec["km"], window)
        print(f"  {label:<26} 试验 {rec['n']:>2} 次 ｜ 清空 {rec['cleared']:>2} 次"
              f"（M2 {rec['cleared'] / rec['n']:>6.1%}）"
              f" ｜ 删失 {rec['censored']:>2} 次（其中提前终止 {rec['early']}"
              f"，占 {rec['early'] / rec['n']:>5.1%}）"
              f" ｜ KM 中位数 {km if km else '> ' + str(window)}")

    # —— 生存曲线落盘：逐轮给风险集、事件数、删失数与 S(t)，可直接在 Excel 画阶梯图
    rows_out = []
    for label, rec, window in summaries:
        if not rec["n"]:
            continue
        med_km = km_median(rec["km"], window)
        for t, risk, d, c, surv in km_curve(rec["km"], window):
            rows_out.append([label, rec["n"], window, t, risk, d, c,
                             f"{surv:.4f}", "是" if med_km == t else "",
                             rec["early"], f"{rec['cleared'] / rec['n']:.4f}",
                             f"{rec['early'] / rec['n']:.4f}"])
    if rows_out:
        km_csv = pathlib.Path(outdir) / "km_curve.csv"
        with open(km_csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["分组", "样本数", "观察窗口", "轮次",
                        "风险集", "该轮清空数", "该轮删失数",
                        "生存率S(t)", "是否KM中位数所在轮",
                        "提前终止数", "M2清空率", "提前终止比例"])
            w.writerows(rows_out)
        print(f"\n  生存曲线已导出：{km_csv}"
              f"（每行一组一行一轮；按“轮次”为横轴、“生存率S(t)”为纵轴做阶梯折线图，"
              f"按“分组”列分系列即得训练内／泛化各自的曲线）")
        # 有 matplotlib 就顺手存一张图；没有不算失败，CSV 已足够在 Excel 出图
        try:
            _plot_km(rows_out, km_csv.with_suffix(".png"))
            print(f"  生存曲线图已导出：{km_csv.with_suffix('.png')}")
        except Exception as e:
            print(f"  （未生成 PNG：{type(e).__name__}；用 Excel 打开 km_curve.csv 画阶梯图即可）")

    print("-" * 78)
    print(f"  M3 口径：窗口内未成功的样本按右删失计入，删失时间取该次试验实际执行到的末轮"
          f"（本批共 {early_stop} 次试验在窗口前提前终止，其中含连续抓空达上限与 LOST）。")
    print("           故 KM 中位数可能大于成功样本的中位数；不得只对成功样本取平均来代表 M3。")
    print("  ⚠ 解释边界：提前终止是「连续抓空达上限」或「LOST」触发的，删失时间与结局相关，"
          "属信息性删失，\n"
          "     不满足随机删失假设。因此 KM 曲线只描述「在本实验的停止策略下，到第 k 轮仍未清空的比例」，\n"
          "     不得解读为「若允许继续执行，本来需要几轮」。报告 M3 时须同时给出各层的提前终止比例与 M2 清空率，\n"
          "     三者缺一即属选择性报告。")
    print("  分层口径：训练内 P1–P4 与泛化 P5／P6 分别统计，不合并；"
          "两层示教覆盖不同，合并会抹掉泛化差距。")
    print("  作废试验（试验状态=作废）已从以上全部汇总中剔除。")
    _m4_report(outdir, train_pos)
    print("=" * 78)
    return 0


def _plot_km(rows, path):
    """按分组画阶梯式生存曲线。缺 matplotlib 时由调用处降级处理，不视为失败。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {}
    for row in rows:                    # 行尾还有提前终止等附加列，只取前 9 列
        label, t, s = row[0], row[3], row[7]
        groups.setdefault(label, []).append((int(t), float(s)))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    max_t = 1
    for i, pts in enumerate(groups.values(), start=1):
        xs, ys = [1], [1.0]
        for t, s in pts:
            xs.extend([t, t + 1])
            ys.extend([s, s])
            max_t = max(max_t, t)
        # 图内只用序号标注：缺中文字体时中文图例会显示成方块，对应关系见 km_curve.csv
        ax.step(xs[:-1], ys[:-1], where="post", label=f"group {i}")
    ax.set_xlabel("round")
    ax.set_ylabel("P(still not cleared)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, max_t + 1)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ======================================================================
# 批量执行：一次启动连续做完一整个位置
# ======================================================================

# 同一位置内按 A→B→A→B… 交错、A 先：既满足「先 A 后 B」，
# 又保留交错以抵消机械臂温升与电池压降的线性漂移。
MIX_ORDER = ["A", "B"]


def group_sequence(count, fixed=None):
    """固定组 -> 重复；MIX -> 按 A→B 交替展开，A 先。"""
    if fixed and fixed != "MIX":
        return [fixed] * count
    return [MIX_ORDER[i % 2] for i in range(count)]


def run_series(args, count):
    """连续做 count 次试验。每次之间停下等人工复位，但不需要重启脚本。"""
    seq = group_sequence(count, args.group)
    pos = args.pos or "P1"
    t0 = args.trial

    print("=" * 66)
    print(f"连续试验    位置 {pos}    共 {count} 次    起始编号 T{t0:02d}")
    for i in range(0, count, 20):
        chunk = seq[i:i + 20]
        print(f"  第 {i + 1:>2}-{i + len(chunk):>2} 次：{' '.join(chunk)}")
    print("  每次之间脚本会停下等你复位方块与收纳区，复位好按回车即可。")
    print("  中途要终止，按 Ctrl+C；已写入的记录不会丢。")
    print("=" * 66)

    done = void = 0
    try:
        for i, g in enumerate(seq):
            args.group = g
            args.trial = t0 + i
            if i:
                cap_i = GROUPS[g]["max_attempts"]
                hint = ("判别器驱动，方块入区即自动收工" if GROUPS[g]["decider"] == "verdict"
                        else "人工监督：每轮需人独立判断是否继续")
                input(f"\n{'─' * 62}\n"
                      f">>> 第 {i + 1}/{count} 次（{g} 组，{hint}）\n"
                      f"    复位 1 个方块、清空收纳区，然后按回车开始本次试验…")
            rc = run_trial(args)
            if rc == 5:
                # 作废试验已落盘留痕，但现场存在未排除的问题，必须停下
                void += 1
                print(f"\n[!] 第 {i + 1} 次已记为「作废」（{args.pos} T{args.trial:02d}）。")
                print("    作废原因见 trials.csv；现场问题排除后，用 --trial 指定同编号重做该次。")
                break
            if rc != 0:
                if rc == 4:
                    print(f"\n[安全] 第 {i + 1} 次停机未确认，已记为「作废」并写入 trials.csv。")
                    print("       请立即断电或按物理急停，确认机械臂完全静止后重新上电归零，"
                          "再用 --trial 指定同编号补做；本批其余试验全部中止。")
                else:
                    print(f"\n[!] 第 {i + 1} 次意外中断（返回码 {rc}），后续试验已停止。")
                    print("    处理完现场后，用 --trial 指定接续编号即可接着做。")
                break
            done += 1
    except KeyboardInterrupt:
        print("\n[!] 已手动终止。")

    print("\n" + "=" * 66)
    print(f"本批完成 {done}/{count} 次" + (f"，作废 {void} 次" if void else "")
          + f"，记录在：{pathlib.Path(args.outdir).resolve()}")
    print("=" * 66)
    return 0


def main():
    ap = argparse.ArgumentParser(description="桌面整理试验执行器（唯一生产入口，A/B 判别开关）")
    ap.add_argument("--group", choices=list(GROUPS) + ["MIX"],
                    help="A=人工监督对照 B=判别器驱动 MIX=按 A→B 交替（A 先）")
    ap.add_argument("--count", type=int, default=None,
                    help="本次连续做几次试验；不给则只做 1 次")
    ap.add_argument("--pos", default=None,
                    help="摆放位置编号。单块只填 1 个（如 P1）。"
                         "P1–P4 记训练内，P5／P6 记泛化（仅诊断），多值写法仅为兼容历史数据")
    ap.add_argument("--trial", type=int, default=1, help="该位置第几次试验（批量时为起始号）")
    ap.add_argument("--manual", action="store_true", help="手动模式：你点 LeLab 按钮")
    ap.add_argument("--max-attempts", type=int, default=None, help="轮数上限，默认按分组")
    ap.add_argument("--retries", type=int, default=None,
                    help="已停用：两组共享 MAX_EMPTY_STREAK=2，传此参数不产生组间差异")
    ap.add_argument("--target", type=int, default=1, help="桌上一共几个方块（本实验固定 1）")
    ap.add_argument("--cam", type=int, default=0, help="判定用相机 index")
    ap.add_argument("--bin", type=lambda s: tuple(int(x) for x in s.split(",")), default=None,
                    help="收纳区 ROI：x,y,w,h；不给则读 bin_roi.json")
    ap.add_argument("--gray", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=(30, 170, 100), help="灰方块：v_lo,v_hi,s_max")
    ap.add_argument("--dark-v", type=int, default=100,
                    help="深色门限：V<=此值即认作方块（不受底色浸染影响）；0=关闭。"
                         "灰块放进黄板后会变成 H10/S190/V45，只靠 --gray 的 S<100 会整块漏掉")
    ap.add_argument("--hsv", type=lambda s: [int(x) for x in s.split(",")],
                    # v_lo 180→140、s_hi 220→255：现场灯光稍暗时块侧面只剩 V140–180，
                    # v_lo=180 会把整块切光（2026-11-01 实测 0 检出，而画面里明明两块）。
                    # 放宽后在 4 个训练集首帧上检出数与旧值完全一致（各 1 块），不引入误检。
                    # 2026-11-08 默认改为 None：当前 1108 布局的目标是**灰方块**，桌上那颗
                    # 红色圆柱是**干扰物**、永远不会移动。开着红色路会把它也当成候选块，
                    # 一旦被 select_blocks 选中，"原位"就画在红柱上，判定必然错（P1_14 实测）。
                    # 只有回到旧 109 数据集（红方块是目标）才手动传 --hsv。
                    default=None,
                    help="彩色方块：h_lo,s_lo,v_lo,h_hi,s_hi,v_hi（默认关闭；旧红块数据集才用）")
    ap.add_argument("--min-area", type=int, default=600)
    ap.add_argument("--keep-ratio", type=float, default=0.40,
                    help="相对占比低于此值算离开原位")
    ap.add_argument("--outdir", default="trial_logs")
    ap.add_argument("--duration", type=int, default=None,
                    help="自动模式每次执行的秒数 = 一轮完整抓放的时长（默认 15 s）")
    ap.add_argument("--policy-ref", default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", default="pick up the block and put it into the box")
    ap.add_argument("--cameras", type=json.loads, default=None,
                    help="JSON，自动模式的相机映射")
    ap.add_argument("--selftest", action="store_true", help="用本机录像验证原位判定")
    ap.add_argument("--calibrate-bin", action="store_true", help="交互框选收纳区并保存")
    ap.add_argument("--derive-a", action="store_true",
                    help="按训练内/泛化分层输出 M2、M3（KM）、M4 汇总与生存曲线")
    ap.add_argument("--export-review", action="store_true",
                    help="把逐轮判定铺成待复核清单 review.csv，供逐轮填人工真值")
    ap.add_argument("--confusion", action="store_true",
                    help="按 review.csv 的人工真值统计在线判别混淆矩阵")
    ap.add_argument("--review-dir", default=None,
                    help="review.csv 所在目录，默认与 --outdir 相同")
    ap.add_argument("--train-pos", default=",".join(TRAIN_POS),
                    help="训练内位置编号，逗号分隔；其余位置一律按泛化统计（默认 P1,P2,P3,P4）")
    args = ap.parse_args()

    if args.cameras is None:
        # 名字必须与模型 config.json 的 input_features 字面对齐（F1 俯视 / F2 腕部）。
        # 填错不会静默成功：lerobot rollout 会抛 Visual feature mismatch；
        # 但两路接反不会报错，只会抓空，务必先对照 check_ready.py 的快照。
        args.cameras = {
            "F1": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30},
            "F2": {"type": "opencv", "camera_index": 1, "width": 640, "height": 480, "fps": 30},
        }

    if args.calibrate_bin:
        return calibrate_bin(args)

    if args.selftest:
        return selftest(args)

    tp = tuple(x.strip() for x in str(args.train_pos).split(",") if x.strip())
    if args.derive_a:
        return derive_a(args.outdir, args.target, cap=(args.max_attempts or 3), train_pos=tp)
    if args.export_review:
        return export_review(args.outdir, tp)
    if args.confusion:
        return confusion(args.review_dir or args.outdir)

    args.bin = load_bin_roi(args.bin)

    # 时长：手动模式不传（由 LeLab 界面决定），自动模式必须显式给或交互确认
    if args.duration is None:
        args.duration = 15 if not args.manual else 0

    if not args.group:
        # 双击进来时用问答方式选开关，避免记参数
        print("=" * 60)
        print("桌面整理试验执行器（唯一生产入口｜单块 / 3 轮 / 60 次）")
        print("  B  = 判别器驱动：成功即停、抓空重试（同为 3 轮预算）… 现行实验默认")
        print("  A  = 人工监督对照：每轮由人独立判断是否继续（机判事后揭晓，须人工值守）")
        print("  MIX= 按 A→B→A→B… 交替、A 先（60 次对照才用）")
        print("=" * 60)
        g = input("选择分组 [A/B/MIX]（回车=B）：").strip().upper() or "B"
        if g not in GROUPS and g != "MIX":
            print("分组无效，退出。")
            return 2
        args.group = g
        if not args.pos:
            args.pos = input("摆放位置编号（单块填 P1／P5 等一个值）：").strip() or "P1"
        if args.count is None:
            c = input("这个位置连续做几次（回车=1）：").strip()
            args.count = int(c) if c.isdigit() and int(c) > 0 else 1
        t = input(f"起始编号（回车={args.trial}）：").strip()
        args.trial = int(t) if t.isdigit() else args.trial
        if not args.manual:
            print()
            print("自动模式需要单次执行的秒数（= 一轮完整抓放的时长，")
            print("不是清空整个桌面的时长；本项目定 15 s，实测后填，不要累加）。")
            d = input("单次执行秒数（回车=15）：").strip()
            args.duration = int(d) if d.isdigit() else 15

    n_pos = len(split_pos(args.pos))
    if args.pos and n_pos != args.target:
        print(f"[!] 位置填了 {n_pos} 个（{args.pos}），桌上 --target {args.target} 个方块。")
        print("    位置只用于记录与分层、不参与控制；但两者数目不一致会让分层结论失真，"
              "先确认是不是漏填了摆放位。")

    if args.count and args.count > 1:
        return run_series(args, args.count)
    return run_trial(args)


if __name__ == "__main__":
    sys.exit(main())
