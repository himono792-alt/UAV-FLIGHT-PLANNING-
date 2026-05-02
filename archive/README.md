# Archive — Phiên bản cũ

Folder lưu các phiên bản cũ của demo và file legacy, để tham chiếu lịch sử và rollback nếu cần.

---

## Files

### `demo_3d_v1.html` — Demo 3D phiên bản đầu tiên

**Đặc điểm:**
- Grid 5×5×3 = 75 cells (nhỏ, demo concept)
- 8 obstacles cố định (1 column + scattered)
- 4 controls cơ bản: A* Search, Bay UAV, Add NFZ, Reset
- Không có battery system, không có toggle panel, không có wind

**Đã thay thế bởi:** `demo_3d.html` ở root repo (phiên bản mở rộng, 12×12×5, có battery + toggles + wind).

**Commit ban đầu:** `e7e2f52 feat(demo): them demo 3D web (Three.js) tai root`

**Khi nào nên xem bản này:**
- Học cách Three.js + A* tích hợp với cấu trúc tối giản
- So sánh để thấy quy trình evolve của project
- Demo nhanh khi chỉ cần concept đơn giản

**Cách mở:**
```bash
# Local
start archive/demo_3d_v1.html       # Windows
open archive/demo_3d_v1.html        # macOS

# Hoặc qua GitHub Pages
https://himono792-alt.github.io/UAV-FLIGHT-PLANNING-/archive/demo_3d_v1.html
```

---

## Lịch sử thay đổi

| Phiên bản | Date | Thay đổi chính |
|---|---|---|
| `demo_3d_v1.html` | 2026-04-11 | Bản đầu tiên — Three.js, 5×5×3 grid, A*, animation cơ bản |
| `demo_3d.html` (current) | 2026-05-02 | Mở rộng 12×12×5, thêm battery gauge SVG, 5 toggle controls, wind slider, random map, fog effect |

---

## Quy ước archive

- File chuyển vào archive khi bị thay thế bởi phiên bản mới
- Đặt tên kèm hậu tố version: `_v1.html`, `_v2.html`...
- Ghi rõ commit hash trong README để dễ tra cứu
- Không xóa file archive trừ khi cần dọn dung lượng
