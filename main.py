#!/usr/bin/env python3
"""
UAV Flight Path Planning Agent - Chương trình chính
=====================================================
Tác tử Lập kế hoạch Đường bay Tối ưu cho UAV

Triết lý thiết kế: ACTING RATIONALLY (Ch1)
    Tác tử luôn chọn hành động tối ưu nhất dựa trên thông tin hiện có
    để đạt mục tiêu: bay đến đích an toàn, nhanh, tiết kiệm năng lượng.

Kiến trúc: LEARNING AGENT (Ch2)
    Performance Element → Thực hiện tìm đường + điều khiển bay
    Critic              → Đánh giá chất lượng đường bay
    Learning Element    → Cập nhật mô hình từ kinh nghiệm
    Problem Generator   → Đề xuất thử nghiệm mới

Luồng xử lý:
    1. PERCEIVE: Sensors thu thập dữ liệu → TELL(KB, percepts)
    2. REASON:   KB kiểm tra an toàn + Bayesian ước tính rủi ro
    3. PLAN:     A*/GA/SA tìm đường tối ưu
    4. ACT:      Thực hiện hành động
    5. LEARN:    Cập nhật mô hình từ kết quả
"""

import os
import sys
import yaml

# Thêm thư mục src vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.grid_world import GridWorld3D
from src.environment.visualizer import Visualizer


def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    """Đọc file cấu hình YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def demo_environment():
    """Demo Giai đoạn 0+1: Tạo và hiển thị môi trường.

    Chạy: python main.py
    """
    print("=" * 60)
    print("  TÁC TỬ LẬP KẾ HOẠCH ĐƯỜNG BAY TỐI ƯU CHO UAV")
    print("  UAV Optimal Flight Path Planning Agent")
    print("=" * 60)
    print()

    # Bước 1: Tạo môi trường từ config
    print("[1] Đang tạo môi trường 3D từ cấu hình...")
    config_path = os.path.join(os.path.dirname(__file__),
                               "configs", "default_config.yaml")
    world = GridWorld3D()
    world.load_from_config(config_path)

    # Hiển thị thông tin
    stats = world.get_stats()
    print(f"    Kích thước: {stats['kích_thước']}")
    print(f"    Start: {stats['start']}")
    print(f"    Goal: {stats['goal']}")
    print(f"    Chướng ngại vật: {stats['chướng_ngại_vật']} ô")
    print(f"    Vùng cấm bay: {stats['vùng_cấm_bay']} ô")
    print(f"    Tỷ lệ chướng ngại: {stats['tỷ_lệ_chướng_ngại']}")
    print(f"    Gió: {stats['gió']}")
    print()

    # Bước 2: PEAS Framework (Ch2)
    print("[2] PEAS Framework (Chương 2 - Intelligent Agents):")
    print("    Performance: Khoảng cách, thời gian, năng lượng, an toàn")
    print("    Environment: 3D, partially observable, stochastic,")
    print("                 sequential, dynamic, continuous")
    print("    Actuators:   Điều khiển hướng bay, tốc độ")
    print("    Sensors:     GPS, LIDAR, cảm biến gió, pin")
    print()

    # Bước 3: Kiểm tra neighbors (Graph-Search foundation - Ch3)
    print("[3] Graph-Search Foundation (Chương 3 - Search):")
    sx, sy, sz = world.start
    neighbors = world.get_neighbors(sx, sy, sz)
    print(f"    Vị trí Start: ({sx}, {sy}, {sz})")
    print(f"    Số ô lân cận có thể đi: {len(neighbors)}")
    print(f"    Các ô lân cận: {neighbors[:5]}...")
    print()

    # Bước 4: Hiển thị 3D
    print("[4] Hiển thị môi trường 3D...")
    print("    (Đóng cửa sổ matplotlib để tiếp tục)")
    print()

    viz = Visualizer(world)
    viz.plot_environment(
        title="Môi trường UAV 3D - Giai đoạn 0: Khởi tạo",
        save_path="assets/environment_demo.png"
    )

    print("=" * 60)
    print("  GIAI ĐOẠN 0 HOÀN THÀNH!")
    print("  Tiếp theo: Giai đoạn 2 - A* Search + Heuristic")
    print("=" * 60)


if __name__ == "__main__":
    demo_environment()
