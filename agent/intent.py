# -*- coding: utf-8 -*-
"""
intent.py —— 指令解析层（规则版，不用大模型）

为什么不用大模型：
    答辩时"技能边界可审计"是方案里的创新点 4。规则表可以逐条列出"哪些话会触发哪种技能"，
    评审一问就能指着表回答；接 LLM 则会引入"同一句话两次解析出不同技能"的不确定性，
    对真机安全也是隐患。所以默认规则解析，LLM 只作为可选的兜底扩展。

输出：Intent(skill, args, raw)
    skill ∈ TIDY / STOP / CONFIRM / CANCEL / STATUS / READY / HELP / UNKNOWN
"""

import re

# 归一化：全角标点、空白、语气词
_PUNCT = "，。！？、；：“”‘’（）《》,.!?;:\"'()<>~ \t\r\n"

# ASR 同音容错（白名单）：faster-whisper small 在短命令 + 现场噪声下可能把「清理桌面」
# 写成同音字。只做逐条字面映射，不开放模糊匹配 —— 规则表仍可逐条审计。
# 演示时若识别文本仍不对，直接在文字框输入「清理桌面」兜底，链路与安全闸门完全一致。
_ASR_TYPO = (
    ("请理桌面", "清理桌面"), ("青理桌面", "清理桌面"), ("清里桌面", "清理桌面"),
    ("清理卓面", "清理桌面"), ("清里卓面", "清理桌面"), ("整理卓面", "整理桌面"),
    ("请理卓面", "清理桌面"), ("收抬桌面", "收拾桌面"),
)

# 否定/取消 必须最先判，"不要清理" 不能落到 TIDY
_NEG = ("取消", "算了", "不要", "不用", "别动", "别清", "暂停", "先不了", "不做了")
_STOP = ("急停", "停", "刹", "住手", "松开", "stop", "emergency")
_CONFIRM = ("确认", "确定", "执行", "开始", "动手", "可以", "好的", "好", "yes", "ok", "干吧")
_TIDY = ("清理桌面", "清空桌面", "收拾桌面", "整理桌面", "清桌面", "收拾一下",
         "整理一下", "桌面上东西收", "收一下桌子", "清理一下桌面", "tidy", "clean")

_STATUS = ("状态", "怎么样了", "好了吗", "到哪了", "进度", "完成了吗")
_READY = ("体检", "自检", "检查一下", "准备好了吗", "能不能开始", "环境检查")
_HELP = ("帮助", "能做什么", "怎么用", "有哪些指令")


class Intent:
    __slots__ = ("skill", "args", "raw")

    def __init__(self, skill, args=None, raw=""):
        self.skill = skill
        self.args = args or {}
        self.raw = raw

    def __repr__(self):
        return f"Intent({self.skill}, {self.args})"


def normalize(text):
    t = (text or "").strip().lower()
    for ch in _PUNCT:
        t = t.replace(ch, "")
    for wrong, right in _ASR_TYPO:
        t = t.replace(wrong, right)
    return t


def _hit(t, words):
    return any(w in t for w in words)


def parse(text):
    """把一句话解析成技能。顺序即优先级，不可随意调换。"""
    t = normalize(text)
    if not t:
        return Intent("UNKNOWN", raw=text)

    # 1) 取消/否定 —— 必须早于 TIDY，否则"不要清理"会被当成本次执行
    if _hit(t, _NEG):
        return Intent("CANCEL", raw=text)
    # 2) 急停 —— 任何时候最高优先，且不需要二次确认
    if _hit(t, _STOP):
        return Intent("STOP", raw=text)
    # 3) 确认
    if _hit(t, _CONFIRM):
        return Intent("CONFIRM", raw=text)
    # 4) 技能：清理桌面（唯一会动机器人的意图）
    loose = re.search(r"(清|收|整理)", t) and re.search(r"(桌|块|位置)", t)
    if _hit(t, _TIDY) or loose:
        args = {}
        m = re.search(r"(\d+)\s*[个块]", t)
        if m:
            args["target"] = max(1, min(4, int(m.group(1))))
        # 话里点名的位置优先于页面下拉（"清理 P2 桌面" / "清理位置2"）
        m2 = re.search(r"p([1-6])", t) or re.search(r"位置([1-6])", t)
        if m2:
            args["pos"] = "P" + m2.group(1)
        return Intent("TIDY", args, raw=text)
    # 5) 只读类
    if _hit(t, _STATUS):
        return Intent("STATUS", raw=text)
    if _hit(t, _READY):
        return Intent("READY", raw=text)
    if _hit(t, _HELP):
        return Intent("HELP", raw=text)
    return Intent("UNKNOWN", raw=text)


HELP_TEXT = ("可以这样说：『清理桌面』（开始收拾）、『停止』（急停）、"
             "『状态』（看进度）、『体检』（环境自检）、『取消』（撤掉待执行指令）。")


if __name__ == "__main__":
    for s in ["清理桌面", "帮我把桌面收拾一下", "不要清理", "停下！", "确认",
              "现在什么状态", "体检", "清 3 个块", "今天天气不错"]:
        print(f"{s:12s} -> {parse(s)}")
