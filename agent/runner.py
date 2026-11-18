# -*- coding: utf-8 -*-
"""
runner.py —— 执行层：把"清理桌面"这个意图，变成一个真的在动的机械臂

职责边界（很重要，别越界）：
    本层只做「拉起 run_trial.py + 收集结果 + 急停」，不重写任何判定逻辑。
    run_trial.py 仍是唯一生产入口，双证据判定、重试上限、终点即停全部由它负责。
    这样语音链路只是"多了一个不按键的触发方式"，实验口径没有被改动，
    答辩时可以直接说：语音层不触碰策略与判别，只替换了人的手指。

设计要点：
    1. 子进程单独进程组：急停时用 taskkill /T 杀整棵树，
       否则 lerobot_rollout 孙进程会继续驱动机械臂（真机事故点）。
    2. 急停顺序固定：先杀本地子进程 → 再调 /stop-inference → 再轮询状态确认。
       反过来会让 run_trial 读到"已空闲"而进入下一轮。
    3. 结果不解析 stdout 中文，只读 trials.csv 的新增行（结构化、口径一致）。
"""

import csv
import json
import os
import re
import pathlib
import subprocess
import threading
import time
import urllib.request

CREATE_NEW_PROCESS_GROUP = 0x00000200
IS_WIN = os.name == "nt"

# 显式禁用系统代理：LeLab 在本机，经代理会返回 502，看起来像服务没起
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 退出码语义（与 run_trial.py 对齐）
EXIT_MEANING = {
    0: ("done", "执行完成"),
    2: ("rejected", "环境/前置条件不满足，未开始执行"),
    3: ("rejected", "画面或摆位不满足，未开始执行"),
    4: ("unsafe", "停机未确认！现场请立即断电或按物理急停"),
    5: ("void", "本次试验作废（取帧/超时等异常），不计入指标"),
}


def _http(host, path, payload=None, timeout=8):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        host + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
    except Exception as e:
        return {"_error": str(e)}


def _inference_active(status):
    """LeLab /inference-status 的真实字段是 inference_active（lelab/rollout.py）。

    早期代码读 active / running —— 两个键都不存在，于是恒判「已停」，
    急停的二次确认变成假阳性：页面显示「已确认机械臂停止」，机械臂可能还在动。
    这里按 inference_active → active → running 取第一个存在的键；
    全都认不出时返回 None（不确定），调用方按「仍在跑」处理，宁严勿松。
    """
    for k in ("inference_active", "active", "running"):
        if k in status:
            return bool(status[k])
    return None


def _line_ts(text, marker, default):
    """取包含 marker 那一行日志里自带的时间戳（LeLab 日志形如
    `INFO 2026-11-08 16:58:31 ...`），解析失败就退回 wall clock。
    用日志自己的时间戳而不是轮询时的 now，是为了让读秒与真实阶段起点对齐。
    """
    for ln in text.splitlines():
        if marker in ln:
            m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", ln)
            if m:
                try:
                    return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    return default
            return default
    return default


class Runner:
    # LeLab 每次 rollout 都写一个新日志（文件名为 unix 秒）。这是唯一能拿到
    # 「策略/硬件准备」与「控制循环真正开始」这两个时刻的可靠信号：
    # /inference-status 只有 inference_active 布尔值，看不出准备与执行的分界。
    INF_LOG_DIR = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"
    # 阶段文案只描述真实发生的事，不做进度假象：
    #   gate    = run_trial 起进程后做门禁与取基准帧，机械臂没动
    #   prep    = LeLab 正在加载策略、连接机械臂与相机，机械臂没动
    #   execute = 日志出现 "Base strategy control loop started"，机械臂开始动作
    #   judge   = 时长到点/rollout 结束，正在取收尾帧与双证据判定，机械臂已停
    PHASE_LABEL = {"gate": "启动与摆位自检中", "prep": "策略与硬件准备中",
                   "execute": "任务执行中", "judge": "收尾判定中"}

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.proc = None
        self.state = {
            "id": 0,
            "state": "idle",          # idle / running / stopping / done / rejected / unsafe / stopped
            "skill": None,
            "cmd": "",
            "message": "",
            "started_at": None,
            "ended_at": None,
            "elapsed": 0,
            "seq": 0,                 # 日志游标，前端按 since 增量取
            "images": [],
            "result": None,
            # —— 真实阶段（供"本轮结果"卡片读秒，不显示假进度）——
            "phase": "",              # gate / prep / execute / judge
            "phase_label": "",
            "phase_since": None,      # 阶段起点（优先取 LeLab 日志自带时间戳）
            "phase_elapsed": 0,
            "round": 0,               # 当前第几轮（每出现一个新 rollout 日志 +1）
            "rounds_cap": 0,
            "duration_s": 0,
            # 早停结论：点早停后「本轮结果」卡片读这一份，全是停止那一刻的真实值。
            # 键：ok（是否已确认停止）/ elapsed（已跑秒数）/ duration（本轮预算）/
            #     round（停在第几轮）/ rounds_cap（轮数上限）/ notes（停止步骤逐条回执）
            "stop_summary": None,
        }
        self.lines = []
        self._base_csv = 0        # 起跑那一刻 trials.csv 的行数（早停时用来判"这轮有没有判过"）

    # ---------------- 对外只读状态 ----------------
    def snapshot(self, since=0):
        with self.lock:
            s = dict(self.state)
            s["elapsed"] = int(time.time() - s["started_at"]) if s["started_at"] and \
                s["state"] in ("running", "stopping") else s.get("elapsed", 0)
            # 阶段读秒：起点取日志时间戳，所以数字是"这一阶段真的过了多久"
            if s.get("phase_since") and s["state"] in ("running", "stopping"):
                s["phase_elapsed"] = max(0, int(time.time() - s["phase_since"]))
            else:
                s["phase_elapsed"] = s.get("phase_elapsed", 0)
            # since < 0 表示「只要游标，不回放历史」：页面每次打开都从空白开始，
            # 不把上一次（乃至上一次启动）的日志再吐一遍。
            if since < 0:
                s["lines"] = []
                s["seq"] = len(self.lines)
                s["baseline"] = True
                return s
            s["lines"] = self.lines[max(0, since):]
            s["seq"] = len(self.lines)
            return s

    def busy(self):
        with self.lock:
            return self.state["state"] in ("running", "stopping")

    # ---------------- 前置检查（不碰相机） ----------------
    def detect_port(self):
        """自动探测从臂串口。串口会随插拔变号，所以每次都探，不写死。"""
        if self.cfg.get("follower_port"):
            return str(self.cfg["follower_port"])
        code = ("import sys;sys.path.insert(0,'.');import check_arm;"
                "p=check_arm.list_ports();print(p[0] if p else '')")
        try:
            out = subprocess.run([self.cfg["python"], "-c", code],
                                 cwd=self.cfg["trial_dir"], capture_output=True,
                                 text=True, timeout=40, encoding="utf-8", errors="replace")
            return out.stdout.strip()
        except Exception:
            return ""

    def preflight(self):
        """只做"能不能起跑"的硬检查。相机占用/曝光交给 run_trial 自己判（它更专业）。"""
        probs = []
        d = pathlib.Path(self.cfg["trial_dir"])
        if not (d / "run_trial.py").exists():
            probs.append(f"找不到 {d / 'run_trial.py'}")
        if not (d / "bin_roi.json").exists():
            probs.append("收纳区未标定（缺 bin_roi.json），先跑 calibrate_bin.bat")
        if not self.cfg.get("policy_ref"):
            probs.append("config.json 里 policy_ref 为空，先填训好的 checkpoint")
        elif os.path.isabs(self.cfg["policy_ref"]) and \
                not pathlib.Path(self.cfg["policy_ref"]).exists():
            probs.append(f"policy_ref 指向的本地模型不存在：{self.cfg['policy_ref']}")
        if not self.detect_port():
            probs.append("没检测到机械臂串口（电源/USB/驱动），自动模式必须给它")
        h = _http(self.cfg["lelab_host"], "/health", timeout=3)
        if h.get("_error"):
            probs.append(f"LeLab 服务不可达（{self.cfg['lelab_host']}），先启动 LeLab")
        else:
            # 第 6 项：LeLab 必须空闲。否则 start-inference 会吃 409，
            # 用户白点一次确认、还白废一次试验记录。
            st = _http(self.cfg["lelab_host"], "/inference-status", timeout=3)
            if not st.get("_error") and _inference_active(st) is True:
                probs.append("LeLab 上已有任务在跑（残留推理 / 遥操作 / 录制），"
                             "先到 LeLab 页面点 Stop 再来")
        return (len(probs) == 0), probs

    # ---------------- 启动一次任务 ----------------
    def start(self, skill, cmd_text, args=None):
        args = args or {}
        if self.busy():
            return False, "机械臂正在执行上一个任务，先等它结束或说『停止』"

        ok, probs = self.preflight()
        if not ok:
            with self.lock:
                self.state.update(state="rejected", message="；".join(probs),
                                  cmd=cmd_text, skill=skill)
            return False, "；".join(probs)

        cfg = self.cfg
        trial_no = self._next_trial_no()
        outdir = cfg.get("outdir") or "trial_logs"
        group = self._pick_group(args.get("group"))
        pos = self._pick_pos(args.get("pos"))
        target = int(args.get("target") or cfg.get("target", 1))
        # 位置个数与方块数不符只提醒不阻断：位置不参与控制，run_trial.py 同样只 warn
        n_pos = len([x for x in str(pos).split("+") if x.strip()])
        if n_pos != target:
            print(f"[提醒] 位置 {pos} 是 {n_pos} 个、桌面 {target} 个方块，分层统计会受影响")
        cmd = [
            cfg["python"], "run_trial.py",
            "--group", group,
            "--pos", pos,
            "--trial", str(trial_no),
            "--max-attempts", str(cfg.get("max_attempts", 3)),
            "--retries", str(cfg.get("retries", 2)),
            "--target", str(target),
            "--cam", str(cfg.get("cam", 0)),
            "--duration", str(cfg.get("duration", 15)),
            "--policy-ref", cfg["policy_ref"],
            "--outdir", outdir,
        ]
        # 绝不传 --count：>1 会让 run_trial 在第二次前 input() 等人工复位，
        # 而子进程没有 stdin，会直接 EOFError 崩掉。批量只能走命令行 run_series。
        # 自动模式这三个参数缺一不可（pydantic 非 Optional），串口每次现探
        cmd += ["--port", self.detect_port(), "--config", str(cfg.get("follower_config") or "test")]
        if cfg.get("task"):
            cmd += ["--task", str(cfg["task"])]
        if cfg.get("cameras"):
            cmd += ["--cameras", json.dumps(cfg["cameras"])]

        base = self._csv_lines(pathlib.Path(cfg["trial_dir"]) / outdir / "trials.csv")
        self._base_csv = base      # 早停时要用它判断本轮是否已经写进 trials.csv

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"     # 不加这行，中文日志在管道里会变 cp936 乱码
        env["PYTHONUNBUFFERED"] = "1"

        with self.lock:
            self.lines = []
            self.state.update(id=trial_no, state="running", skill=skill, cmd=cmd_text,
                              message="机械臂执行中", started_at=time.time(),
                              ended_at=None, elapsed=0, images=[], result=None,
                              phase="gate", phase_label=self.PHASE_LABEL["gate"],
                              phase_since=time.time(), phase_elapsed=0, round=0,
                              rounds_cap=int(cfg.get("max_attempts", 3)),
                              duration_s=int(cfg.get("duration", 15)),
                              stop_summary=None)

        try:
            self.proc = subprocess.Popen(
                cmd, cwd=cfg["trial_dir"], env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=CREATE_NEW_PROCESS_GROUP if IS_WIN else 0)
        except Exception as e:
            with self.lock:
                self.state.update(state="rejected", message=f"启动失败：{e}")
            return False, f"启动失败：{e}"

        threading.Thread(target=self._pump, args=(self.proc, base, trial_no),
                         daemon=True).start()
        threading.Thread(target=self._watch_phase, args=(self.proc, time.time()),
                         daemon=True).start()
        return True, f"已开始（第 {trial_no} 次，分组 {cfg.get('group', 'B')}）"

    # ---------------- 日志泵 + 收尾 ----------------
    def _pump(self, proc, base, trial_no):
        for line in proc.stdout:
            with self.lock:
                self.lines.append(line.rstrip())
        code = proc.wait()
        rec = self._new_trial_record(base, trial_no)
        skill_state, msg = EXIT_MEANING.get(code, ("failed", f"异常退出（code {code}）"))
        if skill_state == "done" and rec:
            # 列名以 run_trial 当前表头为准；旧表头文件会被轮转，这里再兜一层旧列名，
            # 避免表头一变就显示成「共 ? 轮」。
            n_exec = rec.get("A1 实际执行轮数") or rec.get("执行轮数")
            n_show = rec.get("最终入区0-1") or rec.get("进收纳区0-2")
            n_empty = rec.get("抓空次数")
            if str(rec.get("M2 最终入区1/0", "0")).strip() in ("1", "True", "true"):
                msg = (f"已完成：方块入收纳区，第 {rec.get('M3 首次成功轮次', '?')} 轮成功，"
                       f"M4 避免推理 j={rec.get('M4 成功后仍启动推理次数j', '?')}，"
                       f"耗时 {rec.get('A2 耗时s', '?')} 秒")
            else:
                msg = (f"未成功：共 {n_exec if n_exec not in (None, '') else '?'} 轮"
                       f"（抓空 {n_empty if n_empty not in (None, '') else '?'} 次），"
                       f"最终入区 {n_show if n_show not in (None, '') else '?'} 块")
        with self.lock:
            self.state.update(state=skill_state, message=msg, ended_at=time.time(),
                              elapsed=int(time.time() - (self.state["started_at"] or time.time())),
                              result=rec, images=self._images(trial_no),
                              phase="", phase_label="")   # 收尾：清掉阶段，避免残留旧标签
            self.lines.append(f"[总控] {msg}")

    # ---------------- 真实阶段跟踪 ----------------
    def _latest_infer_log(self, t_spawn):
        """取本次运行产生的 LeLab 日志（文件名是 unix 秒，用文件名排序最稳）。"""
        best, best_ts = None, 0
        try:
            for p in self.INF_LOG_DIR.glob("*.log"):
                if not p.stem.isdigit():
                    continue
                ts = int(p.stem)
                if ts < int(t_spawn) - 20:      # 早于本次起跑的一律不看
                    continue
                if ts > best_ts:
                    best, best_ts = p, ts
        except Exception:
            return None
        return best

    def _set_phase(self, code, since=None):
        with self.lock:
            if self.state.get("phase") == code:
                return
            self.state["phase"] = code
            self.state["phase_label"] = self.PHASE_LABEL.get(code, "")
            self.state["phase_since"] = since or time.time()

    def _watch_phase(self, proc, t_spawn):
        """只读 LeLab 日志判断阶段，不改任何执行/判定逻辑。

        判据都是日志里真实写下的事实行：
          Building rollout context            → 开始加载策略、连机械臂与相机（prep）
          Base strategy control loop started  → 控制循环开始，机械臂这一刻才动（execute）
          Duration limit reached / Rollout finished → 本轮结束，转收尾判定（judge）
        每闭一轮（出现新日志文件）轮次 +1，并回到 prep。
        """
        seen = None
        while proc.poll() is None:
            try:
                f = self._latest_infer_log(t_spawn)
                if f is not None:
                    if str(f) != seen:
                        seen = str(f)
                        # 新 rollout：准备阶段严格从这台 rollout 的日志创建时刻算
                        with self.lock:
                            self.state["round"] = self.state.get("round", 0) + 1
                        self._set_phase("prep", f.stat().st_mtime)
                    txt = f.read_text(encoding="utf-8", errors="replace")
                    cur = self.state.get("phase")
                    # 顺序不能反：一旦收了尾，日志里 "control loop started" 那行还在，
                    # 若先判 execute 会把它又改回去，页面上就在「执行中/收尾」之间来回跳。
                    if "Duration limit reached" in txt or "Rollout finished" in txt:
                        if cur in ("prep", "execute"):
                            self._set_phase("judge",
                                            _line_ts(txt, "Duration limit reached", time.time()))
                    elif "Base strategy control loop started" in txt and cur in ("prep", "gate"):
                        self._set_phase("execute",
                                        _line_ts(txt, "Base strategy control loop started",
                                                 time.time()))
            except Exception:
                pass
            time.sleep(0.7)

    # ---------------- 急停 ----------------
    def stop(self):
        """顺序不可换：先杀子进程，再停 LeLab，最后确认状态。"""
        notes = []
        p = self.proc
        was_running = bool(p and p.poll() is None)
        if was_running:
            with self.lock:
                self.state["state"] = "stopping"
            try:
                if IS_WIN:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                   capture_output=True, timeout=15)
                else:
                    p.terminate()
                p.wait(timeout=10)
                notes.append("本地执行进程已终止")
            except Exception as e:
                notes.append(f"终止子进程异常：{e}")
        else:
            notes.append("当前没有正在跑的执行进程")

        host = self.cfg["lelab_host"]
        r = _http(host, "/stop-inference", {}, timeout=10)
        if r.get("_error"):
            # 409 = LeLab 本来就没有 active 推理，这里不是失败，别把它写成"调用失败"去吓人
            if "409" in str(r["_error"]):
                notes.append("推理服务本就空闲（409）")
            else:
                notes.append(f"停止接口调用失败：{r['_error']}")
        confirmed = False
        t0 = time.time()
        while time.time() - t0 < 45:
            s = _http(host, "/inference-status", timeout=3)
            if s.get("_error"):
                notes.append("状态接口不可达，无法确认是否已停")
                break
            if _inference_active(s) is False:
                confirmed = True
                break
            time.sleep(1)
        notes.append("已确认机械臂停止" if confirmed else "未确认停止，请立即断电或按物理急停")

        now = time.time()
        with self.lock:
            st = self.state
            elapsed = int(now - st["started_at"]) if st.get("started_at") else int(st.get("elapsed") or 0)
            trial_no = st.get("id")
            # 这轮判过没有：判过才有"完成/未完成"的资格，没判过只能报"未跑到判定"
            rec = self._record_for_trial(trial_no) if was_running else None
            judged = rec is not None
            task_ok = None
            if judged:
                task_ok = str(rec.get("M2 最终入区1/0", "")).strip() in ("1", "True", "true")
            summary = {
                "ok": bool(confirmed),
                "was_running": was_running,
                "elapsed": elapsed,
                "duration": int(st.get("duration_s") or self.cfg.get("duration") or 0),
                "round": int(st.get("round") or 0),
                "rounds_cap": int(st.get("rounds_cap") or self.cfg.get("max_attempts") or 0),
                "judged": judged,
                "task_ok": task_ok,
                "entered": (rec.get("最终入区0-1") or rec.get("进收纳区0-2")) if judged else None,
                "notes": list(notes),
                "at": now,
                # 没跑到判定 = 现在就补做：线程里跑 judge_now.py，判完把结果写回来
                "judging": bool(was_running and confirmed and not judged),
                "ver": 1,
            }
            msg = ("已早停：" if confirmed else "早停未确认：") + "；".join(notes)
            self.state.update(state="stopped" if confirmed else "unsafe",
                              message=msg, ended_at=now, elapsed=elapsed,
                              stop_summary=summary,
                              # 被 taskkill 后 _pump 收不到尾，这里补一次：磁盘上已经落盘的
                              # 关键帧（起码开局帧）照样给页面，不留空。
                              images=self._images(trial_no) if trial_no else [])
            self.lines.append("[早停] " + "；".join(notes))
        if summary["judging"]:
            # 机械臂已经停稳，立刻补拍一帧把本轮判定做完；
            # 放到线程里跑，别让 /api/stop 这个请求卡在取帧上。
            threading.Thread(target=self._judge_after_stop,
                             args=(trial_no, max(1, int(summary["round"] or 1))),
                             daemon=True).start()
        return confirmed, "；".join(notes)

    def _judge_after_stop(self, trial_no, round_no):
        """早停补判：复用 run_trial 的双证据判据（judge_now.py），回答"这条指令完成没有"。
        结果写回 stop_summary，并把 ver +1 —— 前端靠 ver 变化重画结果卡片。"""
        cfg = self.cfg
        script = pathlib.Path(cfg["trial_dir"]) / "judge_now.py"
        res = {"ok": False, "verdict": "UNKNOWN", "reason": ""}
        if not script.exists():
            res["reason"] = f"缺少 {script.name}，无法补判"
        else:
            cmd = [cfg["python"], str(script), "--trial", str(trial_no),
                   "--pos", self._pick_pos(None), "--round", str(round_no),
                   "--cam", str(int(cfg.get("cam") or 0)),
                   "--target", str(int(cfg.get("target") or 1)),
                   "--outdir", str(cfg.get("outdir") or "trial_logs"), "--json"]
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            try:
                p = subprocess.run(cmd, cwd=cfg["trial_dir"], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace",
                                   timeout=180, env=env)
                lines = [ln for ln in (p.stdout or "").strip().splitlines() if ln.strip().startswith("{")]
                res = json.loads(lines[-1]) if lines else res
                if not lines:
                    res["reason"] = f"补判没有返回结果（exit {p.returncode}）：{(p.stdout or p.stderr or '')[-120:]}"
            except Exception as e:
                res["reason"] = f"补判进程异常：{e}"
        verdict = str(res.get("verdict") or "UNKNOWN")
        ok = bool(res.get("ok")) and verdict in ("SUCCESS", "EMPTY", "LOST")
        with self.lock:
            st = self.state
            if st.get("id") != trial_no or st.get("state") not in ("stopped", "unsafe"):
                return                      # 用户已经起了新一轮，别拿旧结论覆盖
            sm = dict(st.get("stop_summary") or {})
            sm.update(judging=False,
                      judged=ok,
                      task_ok=(verdict == "SUCCESS") if ok else None,
                      verdict=verdict,
                      reason=res.get("reason") or "",
                      bin_gain=res.get("bin_gain"),
                      left_ratio=res.get("left_ratio"),
                      post_frame=res.get("post_frame"),
                      ver=int(sm.get("ver") or 1) + 1)
            st["stop_summary"] = sm
            if ok:
                self.lines.append(f"[判定] {verdict}｜{sm['reason']}")
            else:
                self.lines.append(f"[判定] 补判没做成｜{sm['reason']}")

    # ---------------- 工具 ----------------
    def _pick_group(self, g):
        """只接受单组。MIX 是批量交替（同一位置内 A→B、A 先），单次执行只拿到其中一组，
        放进来会让人以为在做对照。正式 60 次对照走命令行 run_trial.py --count。"""
        g = str(g or self.cfg.get("group") or "B").strip().upper()
        return g if g in ("A", "B") else "B"

    def _pick_pos(self, p):
        """位置编号，单值 P1–P6（本实验桌上只放 1 块）。非法值退回配置默认。"""
        p = str(p or self.cfg.get("pos") or "P1").strip().upper().replace(" ", "")
        keep = [x for x in p.split("+") if re.fullmatch(r"P[1-6]", x)]
        return "+".join(keep) if keep else str(self.cfg.get("pos") or "P1")

    def _csv_lines(self, path):
        try:
            with open(path, encoding="utf-8-sig") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0

    def _new_trial_record(self, base, trial_no=None):
        path = pathlib.Path(self.cfg["trial_dir"]) / (self.cfg.get("outdir") or "trial_logs") / "trials.csv"
        try:
            with open(path, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return None
        if not rows:
            return None
        last = rows[-1]
        # base 是含表头的总行数；正常只有行数变多才是本次新增。
        # 但 run_trial 遇到旧表头会把旧文件轮转重建，行数会变小；
        # 这时改按「试验」编号核对，避免把本次结果整条丢掉。
        if len(rows) >= base:
            return last
        if trial_no is not None and str(last.get("试验", "")).strip() == str(trial_no):
            return last
        return None

    def _record_for_trial(self, trial_no):
        """只认「试验」列等于本轮编号的那一行。
        早停用它回答"这条指令到底完成没有"：有行 = run_trial 已经判完；没行 = 还没跑到判定。
        不能用 _new_trial_record（那边按行数 >= base 会退化成拿上一轮的旧行）。"""
        path = pathlib.Path(self.cfg["trial_dir"]) / (self.cfg.get("outdir") or "trial_logs") / "trials.csv"
        try:
            with open(path, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return None
        if not rows or trial_no is None:
            return None
        last = rows[-1]
        if str(last.get("试验", "")).strip() == str(trial_no):
            return last
        return None

    def _images(self, trial_no):
        out = pathlib.Path(self.cfg["trial_dir"]) / (self.cfg.get("outdir") or "trial_logs")
        pos = self.cfg.get("pos", "P1")
        pats = [f"{pos}_{trial_no:02d}_t0.png"] + \
               [f"{pos}_{trial_no:02d}_a{i}.png" for i in range(1, 9)]
        return [p.name for p in (out / n for n in pats) if p.exists()]

    def _next_trial_no(self):
        f = pathlib.Path(self.cfg.get("_state_file") or "agent_state.json")
        try:
            n = int(json.loads(f.read_text(encoding="utf-8")).get("trial_no", 0)) + 1
        except Exception:
            n = 1
        try:
            f.write_text(json.dumps({"trial_no": n}), encoding="utf-8")
        except Exception:
            pass
        return n
