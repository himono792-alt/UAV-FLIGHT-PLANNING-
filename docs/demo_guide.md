# Hướng dẫn các demo

Dự án có **4 demo** khác nhau, từ CLI đến web 3D trên bản đồ thật, để hiển thị hoạt động của UAV Agent.

| Demo | Mức trực quan | Cần Python | Cần internet | File chính |
|---|---|---|---|---|
| 🌐 3D Web abstract | Cao | Không | Có (CDN) | `demo_3d.html` |
| 🗺️ HCM 3D Map | Rất cao | Tuỳ | Tuỳ | `outputs/hcm_3d_demo.html` (Release) |
| 📊 Visualize matplotlib | Trung | Có | Không | `demo_visualize.py` |
| 💻 CLI | Thấp | Có | Không | `main.py` |

---

## 🌐 Demo 1 — 3D Web abstract (Three.js)

### Truy cập

**Link Pages**: [https://himono792-alt.github.io/UAV-FLIGHT-PLANNING-/demo_3d.html](https://himono792-alt.github.io/UAV-FLIGHT-PLANNING-/demo_3d.html)

Hoặc clone về mở local:
```bash
# Mở thẳng trong browser
start demo_3d.html        # Windows
open demo_3d.html         # macOS
xdg-open demo_3d.html     # Linux
```

### Đặc điểm

- 🎨 UI sci-fi với panel glow xanh dương
- 🖱 Tương tác chuột: kéo để xoay camera, scroll để zoom
- ⌨ Phím tắt: `R` reset camera
- 📊 Stats panel: hiện cost, time, nodes expanded
- 🎯 Có thể chọn algorithm và scenario qua UI

### Yêu cầu

- Browser hỗ trợ WebGL (Chrome, Edge, Firefox, Safari hiện đại)
- Có internet (để load Three.js từ CDN)
- Hoạt động trên cả PC và điện thoại

### Quy ước màu

| Màu | Ý nghĩa |
|---|---|
| 🟥 Đỏ | Obstacle (chướng ngại vật) |
| 🟧 Cam | No-Fly Zone |
| 🟩 Xanh lá | Start (điểm xuất phát) |
| 🟦 Xanh dương | Goal (đích) |
| 🟨 Vàng | Path (đường bay UAV) |

---

## 🗺️ Demo 2 — HCM 3D Map (Three.js + OpenStreetMap)

### Truy cập

Demo này nặng **12 MB** nên không nằm trong git tree. Tải từ GitHub Releases:

**Link**: [Release v1.0-hcm-demo](https://github.com/himono792-alt/UAV-FLIGHT-PLANNING-/releases/tag/v1.0-hcm-demo)

Sau khi tải:

```bash
# Đặt file vào thư mục outputs/ của repo
mkdir -p outputs
mv ~/Downloads/hcm_3d_demo.html outputs/

# Cách 1 — mở thẳng (đủ cho hầu hết tính năng)
start outputs/hcm_3d_demo.html        # Windows
open outputs/hcm_3d_demo.html         # macOS

# Cách 2 — chạy qua static server (khuyến nghị nếu cần fetch tile OSM)
python demo_hcm_3d.py
# Sau đó mở http://127.0.0.1:8765/outputs/hcm_3d_demo.html
```

### Đặc điểm

- 🗺 Bản đồ Quận 1 + Thủ Đức từ OpenStreetMap, building LOD1
- 🏗 Bounding box: `(10.76, 106.69) → (10.8585, 106.7995)` (~12 km × 11 km)
- 🔬 Grid sinh tự động: cell 40 m × 40 m × 10 m, max altitude 260 m
- 🎯 Mission mặc định: PTIT Thủ Đức → Quận 1

### Ba tier so sánh chạy ngay trong trình duyệt

| Tier | Thuật toán | Mục đích |
|---|---|---|
| **1. Search** | BFS, DFS, Greedy BFS, A\* × 3 heuristic, IDA\*, JPS | So sánh 8 thuật toán tìm đường trên cùng map |
| **2. Optimizer** | Simulated Annealing, Genetic Algorithm | Làm mịn lời giải A\* |
| **3. Knowledge** | KB rules (TELL/ASK trực tiếp) | Test phản ứng theo từng rule |

### Tương tác

- 🖱 Drag: xoay camera quanh map
- 🖱 Scroll: zoom in/out
- ⌨ Phím `R`: reset camera
- ☑ Toggle "Nhà dày" / "Lớp bản đồ OSM" / "Show NFZ" trong panel trái
- 💾 Export quỹ đạo: nút "Export GeoJSON" → file giống `outputs/hcm_uav_path.geojson`

### Regenerate demo (nâng cao)

Nếu muốn build lại file `hcm_3d_demo.html` từ dữ liệu OSM mới:

```bash
# Cần cài thêm pyyaml + requests
pip install pyyaml requests

# Nâng chi tiết building từ cache OSM
python tools/enhance_hcm_building_detail.py --config configs/hcm_central.yaml
```

> Script này đọc `configs/hcm_central.yaml` để biết bbox và cell size, sau đó dùng `src/geodata/osm_loader.py` để query Overpass API.

---

## 📊 Demo 3 — Visualize Local (matplotlib 3D)

### Cách chạy

```bash
git clone https://github.com/himono792-alt/UAV-FLIGHT-PLANNING-.git
cd UAV-FLIGHT-PLANNING-
pip install -r requirements.txt
python demo_visualize.py
```

### 4 Scenarios

| # | Scenario | Bạn sẽ thấy |
|---|---|---|
| 1 | **Môi trường trống** | UAV bay đường chéo từ góc → góc đối diện (12×12×5 grid) |
| 2 | **30 obstacles ngẫu nhiên** | UAV luồn lách qua các khối đỏ |
| 3 | **Hẻm hẹp giữa 2 building** | UAV phải bay vòng/leo cao qua "tòa nhà" |
| 4 | **A* vs Greedy BFS** | So sánh 2 thuật toán: A* tìm path optimal, Greedy nhanh hơn nhưng không optimal |

### Tương tác

- 🖱 Kéo chuột trái: xoay 3D
- 🖱 Kéo chuột phải: pan
- 🖱 Scroll: zoom
- ❌ Đóng cửa sổ: chuyển scenario tiếp theo
- 💾 Mọi ảnh tự lưu vào folder `outputs/` (PNG 120 DPI)

### Output mẫu

```
============================================================
  Scenario 2: A* tránh 30 obstacles ngẫu nhiên
============================================================
  Obstacle density: 4.2%
  ✓ A* search
     Path length : 12 bước
     Cost        : 18.83
     Nodes expanded: 256
     Time        : 31.9 ms
```

---

## 💻 Demo 4 — CLI (main.py)

### Cách chạy

```bash
python main.py
```

### 5 Scenarios end-to-end

Demo đầy đủ pipeline UAVAgent: **PERCEIVE → REASON → PLAN → ACT → LEARN**.

| # | Scenario | Tích hợp |
|---|---|---|
| 1 | Bay bình thường (gió 5 m/s, pin 100%) | A*, fly_direct |
| 2 | Nhiều chướng ngại vật (25 obstacles) | A*, DT=AVOID, fly_safe |
| 3 | Gió mạnh (22 m/s) | Bayesian Network + MEU quyết định |
| 4 | Pin yếu 18% → KB→RETURN_HOME | Knowledge Base + FOL |
| 5 | Dynamic re-planning (obstacle giữa đường) | A*, replan, sensor fusion |

### Output mẫu

```
============================================================
  UAV FLIGHT PLANNING AGENT — Demo
  Đồ án TTNT — PTIT
============================================================

Modules tích hợp:
  GĐ1: GridWorld3D + ObstacleManager      (Ch1, Ch2)
  GĐ2: A* Search + heuristics             (Ch3)
  GĐ3: SA + GA optimizer                  (Ch4)
  GĐ4: KB-Agent + Propositional + FOL     (Ch5, Ch6)
  GĐ5: Bayesian Network + MEU             (Ch7, Ch8)
  GĐ6: Decision Tree + Regression + QL    (Ch9)

  ✓ Scenario 1: Bay bình thường (gió 5 m/s, pin 100%)
     [SUCCESS] reached=True | steps=1 | battery_used=23.6% | replans=1 | time=46.6ms

  ✓ Scenario 2: Nhiều chướng ngại vật (25 obstacles)
     [SUCCESS] reached=True | steps=1 | battery_used=23.6% | replans=1 | time=47.6ms

  ✓ Scenario 3: Gió mạnh (22 m/s, BN+MEU quyết định)
     [SUCCESS] reached=True | steps=1 | battery_used=23.6% | replans=1 | time=47.2ms

  ✗ Scenario 4: Pin yếu (18%), KB→RETURN_HOME
     [FAILED] reached=False | steps=1 | battery_used=18.0% | replans=1 | time=44.6ms

  ✓ Scenario 5: Dynamic re-planning (obstacle giữa đường)
     [SUCCESS] reached=True | steps=1 | battery_used=18.9% | replans=1 | time=38.1ms

============================================================
  Kết quả: 4/5 scenarios thành công
============================================================
```

> **Lưu ý**: Scenario 4 báo "FAILED" là **hành vi đúng** — pin yếu nên KB-Agent ra quyết định quay về thay vì cố bay tiếp. `reached=False` ở đây là feature, không phải bug.

---

## So sánh 4 demo

| Demo | Phù hợp | Thời gian setup | Cần internet | Cần Python |
|---|---|---|---|---|
| 🌐 **3D Web abstract** | Show nhanh cho khách, không kỹ thuật | <30s | Có (Pages + CDN) | Không |
| 🗺️ **HCM 3D Map** | Báo cáo, bảo vệ đồ án — trực quan trên bản đồ thật | ~1 phút (tải 12MB) | Tuỳ chế độ | Tuỳ (chỉ cần nếu chạy server) |
| 📊 **Visualize** | Sinh viên/nhà nghiên cứu xem path | 2-3 phút | Không | Có |
| 💻 **CLI** | Test pipeline full, check số liệu | <5s | Không | Có |

---

## Troubleshooting

### Demo 3D Web không load
- Kiểm tra browser hỗ trợ WebGL: vào `chrome://gpu/` xem có "Hardware accelerated"
- Check console (F12) xem có lỗi network/CORS không
- Thử browser khác (Chrome/Edge thường ổn nhất với Three.js)

### Demo HCM 3D Map không hiển thị building / bản đồ trắng
- File 12 MB có thể chưa tải xong — chờ thêm vài giây
- Nếu mở local thẳng (file://) mà thấy lỗi CORS, chạy qua server: `python demo_hcm_3d.py`
- Tắt extension chặn quảng cáo (uBlock, AdBlock) — đôi khi chặn cả OSM tile

### Demo Visualize không mở cửa sổ
- Cài backend GUI: `pip install pyqt5` hoặc dùng `MPLBACKEND=TkAgg python demo_visualize.py`
- Trên server không có display: chỉnh `show=False` trong code, chỉ lưu PNG ra `outputs/`

### main.py báo ImportError
- Đảm bảo đang ở thư mục gốc của repo (`UAV-FLIGHT-PLANNING-/`)
- Đã chạy `pip install -r requirements.txt`

### pytest fail
- Kiểm tra Python version: `python --version` (cần 3.10+)
- Reinstall deps: `pip install -r requirements.txt --upgrade`

---

## Bước tiếp theo

Sau khi xem demo:
- Đọc [Báo cáo đồ án](bao_cao.pdf) — chi tiết phương pháp luận
- Đọc [Mapping lý thuyết](mapping_ly_thuyet.md) — đối chiếu với chương trình PTIT
- Xem source code trong `src/` để hiểu thuật toán
