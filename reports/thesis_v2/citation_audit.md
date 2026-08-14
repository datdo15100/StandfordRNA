# Kiểm tra trích dẫn

## Nguyên tắc

- Mọi nhận định nền tảng về chức năng RNA, kỹ thuật thực nghiệm, folding, TBM, học sâu,
  refinement và metric đều có citation đến paper hoặc nguồn học thuật tương ứng.
- Tiêu đề bài báo trong `references.tex` được giữ nguyên tiếng Anh.
- Kết quả mới của dự án được dẫn về bảng, hình, protocol hoặc artifact nội bộ thay vì gán
  cho một paper bên ngoài.
- Paper tổng kết cuộc thi được ghi rõ là bioRxiv preprint, không mô tả như bài đã peer review.
- Hai nguồn Kaggle được dùng đúng vai trò: documentation chính thức cho protocol và
  notebook công khai cho mốc TBM-only 0.59298.

## Nguồn đã kiểm tra bằng DOI metadata

Các DOI sau đã được đối chiếu với Crossref ngày 2026-08-13:

- 10.1038/s41592-022-01623-y
- 10.1038/s41580-024-00748-6
- 10.1006/jmbi.1999.3001
- 10.1021/ar200098t
- 10.1038/s41392-022-00916-0
- 10.1093/nar/gky949
- 10.1261/rna.031054.111
- 10.1093/nar/gkz1108
- 10.1002/prot.26602
- 10.1093/nargab/lqae048
- 10.3390/molecules28145532
- 10.64898/2025.12.30.696949
- 10.1038/nbt.3988
- 10.1016/0022-2836(70)90057-4
- 10.1016/0022-2836(81)90087-5
- 10.1038/s41467-023-41303-9
- 10.1038/s41592-024-02487-0
- 10.1371/journal.pbio.3003659
- 10.1126/science.abe5650
- 10.1186/s12900-019-0103-1
- 10.1016/j.jmb.2023.168210
- 10.1002/prot.20264
- 10.1038/s41592-022-01585-1
- 10.1093/bioinformatics/btt473
- 10.1107/S0567739476001873

Tác giả, tiêu đề, journal, volume và page hoặc article number trong bibliography được sửa
theo metadata này. Ba entry không dùng DOI là sách bootstrap, trang cuộc thi chính thức
và notebook Kaggle công khai.

## Provenance của số liệu mới

- TBM retrieval và reranking: `../thesis_notes/tbm_whitebox_ablation.md`.
- TBM và DRfold2 complementarity: `../thesis_notes/hybrid_complementarity.md`.
- Geometry refinement confirmatory: báo cáo confirmatory trong `../thesis_notes/`.
- Public và private leaderboard: `../thesis_notes/kaggle_submission_analysis.md`.
- Split 60/20/20 và kết luận khóa: `../thesis_notes/comment_revision_report.md`.

Citation audit tự động xác nhận cả 28 key được dùng trong hai bản luận văn đều có entry
trong `references.tex`.
