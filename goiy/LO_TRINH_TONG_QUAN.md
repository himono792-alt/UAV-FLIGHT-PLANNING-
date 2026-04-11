# LỘ TRÌNH DỰ ÁN UAV FLIGHT PLANNING + GIT WORKFLOW

## Quy tắc Git xuyên suốt
- Mỗi giai đoạn = 1 mốc commit rõ ràng
- Commit message format: `<type>(<scope>): <mô tả>`
- Types: `init`, `feat`, `fix`, `test`, `docs`, `refactor`
- Luôn `git add` từng file/folder cụ thể, KHÔNG `git add .` sau lần đầu
- Push sau mỗi mốc hoàn thành

---

## Tổng quan 9 giai đoạn

| GĐ | Nội dung | Tuần | Branch/Commit | Trạng thái |
|----|----------|------|---------------|------------|
| 0 | Khởi tạo Git + cấu trúc dự án | 1 | `main` — commit init | ✅ ff38981 |
| 1 | Môi trường mô phỏng 3D (Ch2, Ch3) | 2-3 | `feat/environment` | ✅ cb378a7 |
| 2 | A* Search + Heuristic (Ch3, Ch4) | 4-5 | `feat/search` | ✅ 21f1d08 |
| 3 | GA / SA tối ưu hóa (Ch4) | 6-7 | `feat/optimizer` | ✅ pending commit |
| 4 | KB-Agent + Logic Rules (Ch5, Ch6) | 8-9 | `feat/knowledge` | ⬜ |
| 5 | Bayesian Network + MEU (Ch7, Ch8) | 10-11 | `feat/bayesian` | ⬜ |
| 6 | Learning Module (Ch9) | 12-13 | `feat/learning` | ⬜ |
| 7 | Tích hợp + Testing | 14-15 | `feat/integration` | ⬜ |
| 8 | Báo cáo + Bảo vệ | 16 | `docs/report` | ⬜ |

---

## Git Workflow cho mỗi giai đoạn

```bash
# 1. Tạo branch mới
git checkout -b feat/<tên-module>

# 2. Code + test

# 3. Commit theo từng bước nhỏ
git add src/<module>/ tests/test_<module>.py
git commit -m "feat(<module>): <mô tả ngắn>

<chi tiết>"

# 4. Khi xong giai đoạn, merge về main
git checkout main
git merge feat/<tên-module>
git push origin main

# 5. Xóa branch cũ (tùy chọn)
git branch -d feat/<tên-module>
```

---

## File hướng dẫn chi tiết từng giai đoạn

Tất cả nằm trong folder `goiy/`:

- `00_khoi_tao_git_repo.md` — GĐ0: Khởi tạo
- `01_moi_truong_mo_phong.md` — GĐ1: Environment
- `02_a_star_search.md` — GĐ2: Search
- `03_toi_uu_hoa.md` — GĐ3: Optimizer
- `04_kb_agent.md` — GĐ4: Knowledge
- `05_bayesian_meu.md` — GĐ5: Bayesian
- `06_learning.md` — GĐ6: Learning
- `07_tich_hop_testing.md` — GĐ7: Integration
- `08_bao_cao.md` — GĐ8: Report

> Mỗi file sẽ được tạo khi bắt đầu giai đoạn tương ứng.
