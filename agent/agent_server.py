# -*- coding: utf-8 -*-
"""
agent_server.py —— 指令层总控（纯标准库，零第三方依赖）

它做的事只有四件：
    1. 开一个窗口能连的 HTTP 服务（默认 http://localhost:8100）
    2. 收文字 / 收语音 → 解析成技能 → 过安全闸门 → 拉起 run_trial.py
    3. 把机械臂端日志、判定结果、关键帧回推到窗口
    4. 早停：随时能把机械臂停下来

它不做的事（有意为之）：
    不改判定逻辑、不碰策略权重、不重写实验口径。
    run_trial.py 仍是唯一生产入口，本服务只是"替人按了个按钮"。

启动：
    python agent/agent_server.py            # 默认 8100
    python agent/agent_server.py --port 8101
"""

import argparse
import json
import os
import pathlib
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(HERE))
import intent as intent_mod          # noqa: E402
from runner import Runner            # noqa: E402

PENDING_TTL = 30.0                   # 待确认指令的有效期，过期自动作废
CONFIG = {}
RUNNER = None
PENDING = {"skill": None, "args": None, "cmd": "", "at": 0.0}
LOCK = threading.Lock()
ASR = {"model": None, "state": "unloaded", "err": ""}

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

def load_config():
    path = HERE / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg.setdefault("lelab_host", "http://127.0.0.1:8000")
    cfg.setdefault("trial_dir", str(ROOT / "grasp_retry"))
    cfg.setdefault("python", sys.executable)
    cfg.setdefault("group", "B")
    cfg.setdefault("pos", "P1")
    cfg.setdefault("target", 1)
    cfg.setdefault("max_attempts", 3)
    cfg.setdefault("retries", 2)
    cfg.setdefault("cam", 0)
    cfg.setdefault("duration", 280)
    cfg.setdefault("outdir", "trial_logs")
    cfg.setdefault("policy_ref", "")
    cfg.setdefault("require_confirm", True)
    cfg.setdefault("_state_file", str(HERE / "agent_state.json"))
    # 解释器探测顺序：config.json 的 python → 环境变量 LELAB_PY → 当前解释器。
    # 仓库不内置任何本机绝对路径，LeLab 的 venv 请用 LELAB_PY 指过去。
    for cand in (cfg.get("python") or "", os.environ.get("LELAB_PY", "").strip()):
        if cand and pathlib.Path(cand).exists():
            cfg["python"] = cand
            break
    else:
        cfg["python"] = sys.executable
    return cfg


# ----------------------------------------------------------------------
# 语音识别（可选依赖，缺了只影响语音，不影响文字）
# ----------------------------------------------------------------------

ASR = {"model": None, "state": "loading", "err": ""}
ASR_READY = threading.Event()


def preload_asr():
    """后台加载，不阻塞服务启动。失败只影响语音，文字链路照常。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        with LOCK:
            ASR.update(state="missing",
                       err="未安装 faster-whisper。装它才有语音："
                           f'"{sys.executable}" -m pip install faster-whisper')
        ASR_READY.set()
        return
    # 离线优先：答辩现场不依赖 Hugging Face。预置模型放 agent/models/faster-whisper-small/
    # （来源 ModelScope —— 本机 HF 官方超时、hf-mirror 证书被拦）。目录不存在才回退在线下载。
    model_ref = CONFIG.get("asr_model", "small")
    local = HERE / "models" / "faster-whisper-small"
    offline = (local / "model.bin").exists()
    if offline:
        model_ref = str(local)
    try:
        m = WhisperModel(model_ref,
                         device="cpu", compute_type="int8",
                         download_root=str(HERE / "models"))
        with LOCK:
            ASR.update(model=m, state="ready", err="", ref=model_ref, offline=offline)
    except Exception as e:
        with LOCK:
            ASR.update(state="error", err=str(e)[:300])
    finally:
        ASR_READY.set()


def transcribe(raw_bytes, ext="webm"):
    if not ASR_READY.wait(timeout=90):
        return None, "语音模型还在加载，稍等几秒再试"
    if ASR["state"] != "ready":
        return None, ASR["err"]
    tmp = HERE / f"_asr_{int(time.time() * 1000)}.{ext}"
    tmp.write_bytes(raw_bytes)
    try:
        segs, _ = ASR["model"].transcribe(str(tmp), language="zh", beam_size=5,
                                          vad_filter=True)
        return "".join(s.text for s in segs).strip(), ""
    except Exception as e:
        return None, str(e)[:300]
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ----------------------------------------------------------------------
# 指令处理（安全闸门都在这里）
# ----------------------------------------------------------------------

def status_text():
    s = RUNNER.snapshot()
    # 运行中要说清"现在到底在哪一步、这一步过了多久"，否则只说"已跑 N 秒"
    # 会让人在准备阶段误以为机械臂已经在动。
    if s["state"] in ("running", "stopping"):
        ph = s.get("phase_label") or ""
        if not ph:
            return f"执行中，总时长 {s['elapsed']} 秒"
        tail = f"（{ph} {s.get('phase_elapsed', 0)} 秒"
        if s.get("round"):
            tail += f"，第 {s['round']}/{s['rounds_cap']} 轮"
        tail += "）"
        return f"执行中，总时长 {s['elapsed']} 秒" + tail
    return {
        "idle": "空闲，等你说『清理灰色方块』",
        "done": s.get("message") or "上一轮已完成",
        "rejected": s.get("message") or "上一轮未开始",
        "unsafe": s.get("message") or "停机未确认，请断电",
        "stopped": s.get("message") or "已停止",
    }.get(s["state"], s["state"])


def handle_command(text, source="text", params=None):
    """返回 (http_code, payload)。所有会动机器人的路径都必须经过这里。"""
    it = intent_mod.parse(text)
    say, need_confirm, ok = "", False, True

    if it.skill == "STOP":
        ok2, msg = RUNNER.stop()
        say = "已早停。" if ok2 else f"早停未确认：{msg}"
        ok = ok2
        with LOCK:
            PENDING.update(skill=None, at=0.0)

    elif it.skill == "CANCEL":
        with LOCK:
            had = PENDING["skill"] is not None
            PENDING.update(skill=None, args=None, cmd="", at=0.0)
        say = "已取消待执行指令" if had else "没有待执行的指令"

    elif it.skill == "CONFIRM":
        with LOCK:
            p = dict(PENDING)
            PENDING.update(skill=None, args=None, cmd="", at=0.0)
        if p["skill"] and (time.time() - p["at"]) <= PENDING_TTL:
            ok, say = RUNNER.start(p["skill"], p["cmd"], p["args"])
        else:
            ok, say = False, "没有待确认的指令（或已超时），请先说要做什么"

    elif it.skill == "TIDY":
        # 页面已不暴露分组 / 位置 / 块数；这里只认话里点名的（"清理 P2"）与 config.json
        arg = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
        arg.update(it.args)
        g = str(arg.get("group") or CONFIG.get("group", "B")).strip().upper()
        if g == "MIX":
            ok, say = False, ("单次执行不能选 MIX——MIX 是同一位置内 A→B 交替（A 先）的批量排程，"
                              "只跑一次拿不到对照。现行实验固定 B 组；"
                              "如需 A/B 对照请走命令行 run_trial.py --count。")
        elif RUNNER.busy():
            ok, say = False, "机械臂正在执行上一个任务，先等它结束或说『停止』"
        else:
            # 先体检再进确认：让环境问题在"要不要动手"这一步就暴露，而不是等点了确认才报错
            ready, probs = RUNNER.preflight()
            if not ready:
                ok, say = False, "现在不能开始：" + "；".join(probs)
            elif CONFIG.get("require_confirm", True):
                with LOCK:
                    PENDING.update(skill="TIDY", args=arg, cmd=text, at=time.time())
                need_confirm = True
                # 颜色在位预检接进来之后，这里改成：
                #   收到【清理灰色方块】指令，识别到【灰色方块】，开始清理，确认吗？
                say = (f"收到【{text}】指令，开始清理，确认吗？"
                       f"{PENDING_TTL:.0f} 秒内说『确认』或点确认。")
            else:
                ok, say = RUNNER.start("TIDY", text, arg)

    elif it.skill == "STATUS":
        say = status_text()

    elif it.skill == "READY":
        ok2, probs = RUNNER.preflight()
        ok = ok2
        say = "环境就绪，可以开始。" if ok2 else "还不能开始：" + "；".join(probs)

    else:
        ok = False
        say = intent_mod.HELP_TEXT

    return (200 if ok else 400), {
        "ok": ok, "say": say, "skill": it.skill, "need_confirm": need_confirm,
        "pending_ttl": PENDING_TTL if need_confirm else 0,
        "state": RUNNER.snapshot()["state"] if RUNNER else "idle",
        "text": text, "source": source, "asr": ASR["state"],
    }


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "TidyAgent/1.0"

    def log_message(self, fmt, *a):      # 静音访问日志，只留业务日志
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        if p in ("/", "/index.html"):
            f = HERE / "ui.html"
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        if p == "/api/state":
            q = parse_qs(u.query)
            return self._send(200, RUNNER.snapshot(int(q.get("since", ["0"])[0])))
        if p == "/api/asr_status":
            return self._send(200, {"state": ASR["state"], "err": ASR["err"],
                                    "model": ASR.get("ref") or CONFIG.get("asr_model", "small"),
                                    "offline": bool(ASR.get("offline", False))})
        if p.startswith("/files/"):
            name = os.path.basename(p[len("/files/"):])
            out = pathlib.Path(CONFIG["trial_dir"]) / CONFIG.get("outdir", "trial_logs") / name
            if out.exists() and out.is_file():
                return self._send(200, out.read_bytes(), "image/png")
            roi = pathlib.Path(CONFIG["trial_dir"]) / name
            if roi.exists() and roi.is_file() and name == "bin_roi_preview.png":
                return self._send(200, roi.read_bytes(), "image/png")
            return self._send(404, {"error": "no such image"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        p = u.path

        if p == "/api/cmd":
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                return self._send(400, {"ok": False, "say": "请求格式不对"})
            params = {k: body.get(k) for k in ("group", "pos", "target")}
            code, payload = handle_command(body.get("text", ""), "text", params)
            return self._send(code, payload)

        if p == "/api/asr":
            if not raw:
                return self._send(400, {"ok": False, "say": "没收到音频"})
            ext = self.headers.get("X-Audio-Ext", "webm")
            text, err = transcribe(raw, ext)
            if text is None:
                return self._send(503, {"ok": False, "say": f"语音识别不可用：{err}",
                                        "asr": ASR["state"]})
            if not text:
                return self._send(200, {"ok": False, "say": "没听清，再说一次？",
                                        "text": "", "asr": "ready"})
            params = {k: self.headers.get(f"X-{k.capitalize()}") for k in ("group", "pos")}
            if self.headers.get("X-Target"):
                params["target"] = self.headers["X-Target"]
            code, payload = handle_command(text, "voice", params)
            payload["text"] = text
            return self._send(code, payload)

        if p == "/api/confirm":
            code, payload = handle_command("确认", "button")
            return self._send(code, payload)

        if p == "/api/cancel":
            code, payload = handle_command("取消", "button")
            return self._send(code, payload)

        if p == "/api/stop":
            code, payload = handle_command("停止", "button")
            return self._send(code, payload)

        return self._send(404, {"error": "not found"})


def main():
    global CONFIG, RUNNER
    ap = argparse.ArgumentParser(description="桌面清理 语音/文字指令层")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-confirm", action="store_true", help="跳过二次确认（仅演示用）")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    ap.add_argument("--outdir", default=None,
                    help="覆盖 trial_logs 子目录；答辩演示传 demo_logs，数据不污染正式 60 次")
    a = ap.parse_args()

    CONFIG = load_config()
    if a.outdir:
        CONFIG["outdir"] = a.outdir
    if a.no_confirm:
        CONFIG["require_confirm"] = False
    RUNNER = Runner(CONFIG)

    ok, probs = RUNNER.preflight()
    print("=" * 62)
    print("  桌面清理 · 指令层已启动")
    print(f"  窗口地址 : http://{a.host}:{a.port}")
    print(f"  试验目录 : {CONFIG['trial_dir']}")
    print(f"  LeLab    : {CONFIG['lelab_host']}")
    print(f"  执行分组 : {CONFIG['group']}   位置 {CONFIG['pos']}   轮数上限 {CONFIG['max_attempts']}")
    print(f"  二次确认 : {'开' if CONFIG['require_confirm'] else '关（--no-confirm）'}")
    print(f"  数据目录 : {CONFIG.get('outdir', 'trial_logs')}")
    print(f"  前置检查 : {'全部通过' if ok else '未通过 → ' + '；'.join(probs)}")
    print("=" * 62)

    threading.Thread(target=preload_asr, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    if a.open:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出。若机械臂还在动，请说『停止』或直接断电。")


if __name__ == "__main__":
    main()
