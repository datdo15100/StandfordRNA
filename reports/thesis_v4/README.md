# Thesis V4 working draft

V4 là bản thảo mới, không sửa đè V3. Scientific spine là phân rã pipeline theo objective
TM-score: TBM, candidate-source allocation, refinement và complete-system external check.

## Evidence state

- Preregistration frozen: commit `7739ff6`, tag `v4-preregistration-2026-08-20`.
- Master train-V2 ledger: 5.135 RNA; CASP15 nằm trong development ledger riêng.
- P0-production reproduction: PASS, numeric maximum absolute error 0 cho cả hai priors.
- Final-test native performance: chưa mở.
- Results có `Working-draft evidence gate` là shell nội bộ, không phải kết luận và phải
  được thay bằng frozen V4 evidence trước bản thesis cuối.

V3 chỉ được dùng làm reference cho giọng viết, kiến thức nền và citation còn đúng. Không
số liệu V3 nào được tự động chuyển vào Results V4.

## Build

Tài liệu dùng XeLaTeX và Times New Roman. Trong WSL, preamble có fallback tới font
Windows ở `/mnt/c/Windows/Fonts`. File tài liệu tham khảo hiện dùng lại bibliography đã
audit của V3 bằng relative input; citation provenance được audit lại khi method section
được freeze.

```text
cd reports/thesis_v4
xelatex thesis_vi.tex
xelatex thesis_vi.tex
```
