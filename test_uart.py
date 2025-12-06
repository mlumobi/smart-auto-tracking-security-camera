#!/usr/bin/env python3
"""
测试 UART 通信 - 手动发送舵机指令
用于诊断舵机是否响应
"""

import serial
import time

try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
    print("✓ UART 连接成功")
    time.sleep(2)
    
    print("\n测试序列开始...")
    
    # 测试1: 归中
    print("1. 归中位置 (90, 90)")
    ser.write("pan=90\n".encode())
    ser.write("tilt=90\n".encode())
    time.sleep(2)
    
    # 测试2: 左移
    print("2. Pan 左移到 60")
    ser.write("pan=60\n".encode())
    time.sleep(2)
    
    # 测试3: 右移
    print("3. Pan 右移到 120")
    ser.write("pan=120\n".encode())
    time.sleep(2)
    
    # 测试4: 归中
    print("4. Pan 归中到 90")
    ser.write("pan=90\n".encode())
    time.sleep(2)
    
    # 测试5: Tilt 上
    print("5. Tilt 上移到 60")
    ser.write("tilt=60\n".encode())
    time.sleep(2)
    
    # 测试6: Tilt 下
    print("6. Tilt 下移到 120")
    ser.write("tilt=120\n".encode())
    time.sleep(2)
    
    # 测试7: 归中
    print("7. 全部归中 (90, 90)")
    ser.write("pan=90\n".encode())
    ser.write("tilt=90\n".encode())
    time.sleep(1)
    
    print("\n✓ 测试完成")
    print("如果舵机没有移动，请检查：")
    print("  1. ESP32 串口连接是否正确")
    print("  2. ESP32 是否已上传正确的程序")
    print("  3. 舵机电源是否连接")
    print("  4. 串口设备是否是 /dev/serial0")
    
    # 读取 ESP32 的调试输出
    print("\n等待 ESP32 响应...")
    time.sleep(0.5)
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"  ESP32: {line}")
    
    ser.close()
    
except Exception as e:
    print(f"✗ 错误: {e}")
    print("\n可能的原因：")
    print("  1. UART 未启用 (需要在 /boot/config.txt 中启用)")
    print("  2. 权限不足 (需要 sudo 或添加到 dialout 组)")
    print("  3. ESP32 未连接")

