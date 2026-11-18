#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收纳区标定（tkinter 版）。

为什么不用 cv2.selectROI：
    LeLab 的 venv 装的是 opencv-python-headless，不带 highgui，
    cv2.imshow / cv2.selectROI 一律抛 "The function is not implemented"。
    这里改用 tkinter 画布显示同一帧，鼠标拖矩形，回车保存，
    输出格式与 run_trial.py --calibrate-bin 完全一致。

用法：
    python calibrate_gui.py              # 默认相机 0
    python calibrate_gui.py --cam 1

输出：
    bin_roi.json          {"bin": [x, y, w, h], "saved_at": "..."}
    bin_roi_preview.png   带红框的预览图，用于人工核对
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import cv2
import tkinter as tk

HERE = pathlib.Path(__file__).resolve().parent
BIN_JSON = HERE / "bin_roi.json"
PREVIEW = HERE / "bin_roi_preview.png"
W, H = 640, 480


def grab(cam: int, tries: int = 8, gap: float = 0.8):
    """取一帧；相机被上一进程占用时 Windows 释放 DSHOW 句柄有延迟，故重试。"""
    for _ in range(tries):
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
                frame = None
                for _ in range(8):          # 丢弃前几帧，等自动曝光稳定
                    ok, tmp = cap.read()
                    if ok:
                        frame = tmp
            finally:
                cap.release()
            if frame is not None:
                return frame
        else:
            cap.release()
        time.sleep(gap)
    return None


def to_ppm(img_bgr) -> bytes:
    """numpy BGR -> PPM(P6) 字节串；tkinter 的 PhotoImage 原生支持 PPM。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()


def save(roi, frame) -> None:
    x, y, w, h = roi
    BIN_JSON.write_text(json.dumps({"bin": [x, y, w, h],
                                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    prev = frame.copy()
    cv2.rectangle(prev, (x, y), (x + w, y + h), (0, 0, 255), 2)
    ok, buf = cv2.imencode(".png", prev)      # 中文路径下 imwrite 会静默失败
    if ok:
        PREVIEW.write_bytes(buf.tobytes())


class Picker:
    def __init__(self, root: tk.Tk, frame, cam: int = 0):
        self.root = root
        self.frame = frame
        self.CAM = cam
        self.x0 = self.y0 = self.x1 = self.y1 = None
        self.rect_id = None
        self.result = None

        root.title("框选收纳区 —— 拖一个矩形，回车保存")
        root.resizable(False, False)

        self.photo = tk.PhotoImage(data=to_ppm(frame), format="ppm")
        self.canvas = tk.Canvas(root, width=self.photo.width(),
                                height=self.photo.height(), cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.hint = tk.Label(root, text="拖出矩形框住【收纳区内部】，越小越安全；回车保存，R 重取，Esc 取消",
                             font=("Microsoft YaHei", 9))
        self.hint.pack(fill="x")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        root.bind("<Return>", lambda e: self.confirm())
        root.bind("<Escape>", lambda e: self.cancel())
        root.bind("<r>", lambda e: self.refresh())
        root.bind("<R>", lambda e: self.refresh())

    # --- 鼠标 ---
    def on_press(self, e):
        self.x0, self.y0 = e.x, e.y
        self.x1, self.y1 = e.x, e.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                    outline="#00e000", width=2)

    def on_drag(self, e):
        if self.x0 is None:
            return
        self.x1, self.y1 = e.x, e.y
        self.canvas.coords(self.rect_id, self.x0, self.y0, e.x, e.y)
        self.hint.config(text=f"当前 {abs(e.x - self.x0)} × {abs(e.y - self.y0)} px"
                              f"　回车保存，R 重取，Esc 取消")

    def on_release(self, e):
        self.x1, self.y1 = e.x, e.y
        w, h = abs(self.x1 - self.x0), abs(self.y1 - self.y0)
        if w < 20 or h < 20:
            self.hint.config(text="框太小（<20px），重拖一次")
            self.x0 = None

    # --- 键盘 ---
    def roi(self):
        if self.x0 is None or self.x1 is None:
            return None
        x = min(self.x0, self.x1)
        y = min(self.y0, self.y1)
        w = abs(self.x1 - self.x0)
        h = abs(self.y1 - self.y0)
        if w < 20 or h < 20:
            return None
        return (int(x), int(y), int(w), int(h))

    def confirm(self):
        r = self.roi()
        if r is None:
            self.hint.config(text="还没框出有效矩形（至少 20×20）")
            return
        self.result = r
        self.root.destroy()

    def cancel(self):
        self.result = None
        self.root.destroy()

    def refresh(self):
        f = grab(self.CAM)
        if f is None:
            self.hint.config(text="取帧失败，相机可能被占用")
            return
        self.frame = f
        self.photo = tk.PhotoImage(data=to_ppm(f), format="ppm")
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.rect_id = None
        self.x0 = self.y0 = None
        self.hint.config(text="已重取一帧，重新拖框")


def main() -> int:
    ap = argparse.ArgumentParser(description="收纳区标定（tkinter 版）")
    ap.add_argument("--cam", type=int, default=0, help="相机 index，默认 0")
    args = ap.parse_args()

    frame = grab(args.cam)
    if frame is None:
        print(f"[!] 相机 {args.cam} 打不开。先确认 LeLab 已关闭，没占着相机。")
        return 2

    print("即将打开取景窗口：用鼠标拖一个矩形，框住【收纳区内部】。")
    print("  ① 框内不要含收纳区外沿；② 宁可比收纳区小一点，不要框进桌面；")
    print("  ③ 看好后按回车保存，R 重取一帧，Esc 取消。\n")

    root = tk.Tk()
    ui = Picker(root, frame, cam=args.cam)
    root.mainloop()

    if ui.result is None:
        print("[!] 已取消，未写入任何文件。")
        return 2

    save(ui.result, ui.frame)
    print(f"\n已保存收纳区 {ui.result[0]},{ui.result[1]},{ui.result[2]},{ui.result[3]} -> {BIN_JSON}")
    print(f"预览图：{PREVIEW}（请打开确认红框正好套住收纳区）")
    print("\n下一步：摆 1 个方块到桌面、清空收纳区，再跑 run_trial_local.bat 做真机联调。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
