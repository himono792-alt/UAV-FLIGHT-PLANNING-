# Tác tử Lập kế hoạch Đường bay Tối ưu cho UAV
## UAV Optimal Flight Path Planning Agent

**Đồ án môn Trí tuệ Nhân tạo - Học viện Công nghệ Bưu chính Viễn thông (PTIT)**

## Mô tả

Xây dựng một Tác tử thông minh (Intelligent Agent) có khả năng lập kế hoạch 
đường bay tối ưu cho UAV trong môi trường 3D có chướng ngại vật, điều kiện 
thời tiết thay đổi, và ràng buộc về năng lượng.

Dự án tích hợp toàn diện **9/9 chương** lý thuyết Trí tuệ Nhân tạo:

| Chương | Lý thuyết | Module |
|--------|-----------|--------|
| Ch1 | Introduction to AI | Triết lý thiết kế: Acting Rationally |
| Ch2 | Intelligent Agents | PEAS Framework, Learning Agent Architecture |
| Ch3 | Solving Problems by Searching | Graph-Search, State Space |
| Ch4 | Informed Search | A* Search, Heuristic, GA, SA |
| Ch5 | Logical Agent | KB-Agent, TELL/ASK, Propositional Logic |
| Ch6 | First-Order Logic | FOL Rules, Predicates, Quantifiers |
| Ch7 | Quantifying Uncertainty | Bayes' Rule, MEU, Decision Theory |
| Ch8 | Probabilistic Reasoning | Bayesian Network, CPT, Inference |
| Ch9 | Learning from Examples | Decision Tree, Regression, Q-Learning |

## Kiến trúc hệ thống

```
SENSORS → KNOWLEDGE BASE → DECISION ENGINE → LEARNING → ACTUATORS
  GPS       FOL Rules         A* Search        Critic     Motor
  LIDAR     Logic Rules       GA/SA            DT/RL      Direction
  Wind      Bayesian Net      MEU              Update     Speed
```

## Cài đặt

```bash
# Clone repo
git clone https://github.com/USERNAME/uav-flight-planning.git
cd uav-flight-planning

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

## Chạy

```bash
# Chạy chương trình chính
python main.py

# Chạy tests
python -m pytest tests/ -v
```

## Cấu trúc dự án

```
uav-flight-planning/
├── src/
│   ├── environment/    # Môi trường mô phỏng 3D (Ch2, Ch3)
│   ├── search/         # Thuật toán tìm kiếm (Ch3, Ch4)
│   ├── optimizer/      # GA, SA tối ưu hóa (Ch4)
│   ├── knowledge/      # KB-Agent, Logic (Ch5, Ch6)
│   ├── bayesian/       # Bayesian Network, MEU (Ch7, Ch8)
│   ├── learning/       # Machine Learning (Ch9)
│   └── agent/          # Tích hợp UAV Agent
├── tests/              # Unit tests
├── configs/            # File cấu hình
├── docs/               # Tài liệu, báo cáo
└── assets/             # Bản đồ, hình ảnh
```

## GVHD

**Cô Hai Thị Tuyết Nguyên** - Học viện Công nghệ Bưu chính Viễn thông
