"""排查 SO-101 跟随臂连接：列出串口、扫描电机 ID、指出缺失的 ID。

用法（双击 check_arm.bat 或命令行）：
    python check_arm.py            # 扫描所有串口
    python check_arm.py COM10      # 只扫指定串口

判读：
    6/6 电机都在           → 硬件正常，可以直接跑推理/遥操作
    缺 6 号（夹爪）         → 最常见：夹爪堵转触发舵机过载保护、或末端排线松
    一个电机都没有          → USB 没连上 / 舵机没供电 / 串口被别的程序占用
"""

from __future__ import annotations

import sys
import time

EXPECTED_IDS = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as lp

        return sorted(p.device for p in lp.comports())
    except Exception as exc:  # pyserial 缺失时退回注册表
        print(f"[warn] pyserial 不可用（{exc}），改用注册表枚举")
        import winreg

        ports: list[str] = []
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
            i = 0
            while True:
                try:
                    _, value, _ = winreg.EnumValue(key, i)
                except OSError:
                    break
                ports.append(value)
                i += 1
        except OSError:
            pass
        return sorted(ports)


def scan(port: str) -> dict[int, list[int]]:
    from lerobot.motors.feetech.feetech import FeetechMotorsBus

    return FeetechMotorsBus.scan_port(port)


def main() -> int:
    wanted = sys.argv[1].upper() if len(sys.argv) > 1 else None
    ports = list_ports()
    if wanted:
        ports = [p for p in ports if p.upper() == wanted] or [wanted]

    print("=" * 62)
    print(" SO-101 串口与电机自检")
    print("=" * 62)
    print(f"系统串口：{', '.join(ports) if ports else '（一个都没有）'}")
    print()

    if not ports:
        print("[结论] 系统里没有任何串口。")
        print("  1) 检查两臂 USB 线是否插紧，换一个 USB 口试试")
        print("  2) 检查舵机电源（12V 适配器）是否上电")
        print("  3) 若刚跑过推理，先断电 10 秒给舵机复位，再上电重插")
        return 2

    ok_ports: list[tuple[str, dict[int, list[int]]]] = []
    for port in ports:
        print(f"--- 扫描 {port} ---")
        try:
            found = scan(port)
        except Exception as exc:
            msg = str(exc).splitlines()[0][:120]
            print(f"  打不开或被占用：{msg}")
            print("  → 关掉 LeLab（lelab-zh.exe --stop）后再试")
            continue
        if not found:
            print("  没扫到任何电机（该口可能是另一个臂的调试口，或没接电机）")
            continue
        for baud, ids in found.items():
            print(f"  波特率 {baud}: 找到 ID {sorted(ids)}")
        best = max(found.items(), key=lambda kv: len(kv[1]))
        ok_ports.append((port, found))
        print(f"  → {port} 上共 {len(best[1])} 个电机，ID {sorted(best[1])}")
        print()

    if not ok_ports:
        print("[结论] 所有串口都没扫到电机。")
        print("  1) 舵机电源是否上电（12V）？电机板指示灯是否亮？")
        print("  2) USB 线是否插在舵机板的 USB 口（不是充电口）？")
        print("  3) 设备管理器里是否出现 CH343 串口？")
        return 3

    print("=" * 62)
    rc = 0
    for port, found in ok_ports:
        ids = set(max(found.items(), key=lambda kv: len(kv[1]))[1])
        missing = sorted(set(EXPECTED_IDS) - ids)
        if not missing:
            print(f"[{port}] 6/6 电机齐全 ✔ 硬件正常")
            continue
        rc = 1
        names = "、".join(f"{i}={EXPECTED_IDS[i]}" for i in missing)
        print(f"[{port}] 缺少 {len(missing)} 个电机：{names}")
        if missing == [6]:
            print("  → 典型是夹爪堵转/过载触发了舵机保护：")
            print("     1) 拔掉舵机电源 10 秒再上电（保护锁存必须断电才复位）")
            print("     2) 把夹爪手动掰开到张开位置，别让它顶着东西")
            print("     3) 上电后先做一次遥操作，确认夹爪能开合，再跑推理")
            print("     4) 反复出现则检查 6 号舵机的串联排线（末端最易松）")
        else:
            print("  → 先断电重启，再检查对应舵机的串联排线")
    print("=" * 62)
    return rc


if __name__ == "__main__":
    t0 = time.time()
    try:
        code = main()
    except KeyboardInterrupt:
        code = 130
    print(f"耗时 {time.time() - t0:.1f}s")
    sys.exit(code)
