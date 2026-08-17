# Thesis v3

Thư mục này chứa bản thảo v3 bằng tiếng Việt và tiếng Anh. Hai bản dùng cùng cấu trúc,
số liệu, hình và tài liệu tham khảo. Font chính là Times New Roman; mọi kết quả được làm
tròn đến ba chữ số sau dấu phẩy.

## Phạm vi của v3

V3 trình bày rõ pipeline được kế thừa từ hướng thắng cuộc của John và tách phần luận văn
thay đổi:

1. TBM an toàn thời gian với MMseqs2, composite search, reranking và gap filling.
2. Hai candidate DRfold2 được thêm vào ba candidate TBM để tăng độ đa dạng nguồn.
3. Geometry Refinement sửa chất lượng cục bộ nhưng không được claim làm tăng TM-score.

Boltz không nằm trong đường chạy cuối của luận văn.

## Kết quả chính

- TBM tái lập từ John đạt 0.298 trên 12 RNA CASP15; TBM của luận văn đạt 0.307.
- Trên 20 RNA validation, bank 3T+2D đạt 0.588, cao hơn 3T 0.023.
- Geometry Refinement tăng C1'-lDDT 0.007, giảm SW-RMSD9 0.040 angstrom và làm
  TM-score thay đổi -0.001.
- Pipeline cuối đạt 0.628 public và 0.614 private trên Kaggle.
- Điểm private cuối cao hơn notebook TBM-only của John 0.021, notebook hybrid công khai
  0.038 và đội thắng cuộc 0.036.

Hai submission của luận văn không phải ablation một biến. Điểm chênh trên Kaggle chỉ được
gọi là tác động toàn pipeline. So sánh trực tiếp với refiner rule-based của John và
factorial private 2 x 2 vẫn là hai thí nghiệm cần hoàn thành trước khi khóa luận văn.

## Tệp chính

- `thesis_vi.tex`, `thesis_vi.pdf`: bản tiếng Việt, 54 trang.
- `thesis_en.tex`, `thesis_en.pdf`: bản tiếng Anh, 53 trang.
- `references.tex`: tài liệu tham khảo dùng chung.
- `figures/`: hình PDF vector và PNG.
- `citation_audit.md`: kiểm tra nguồn trích dẫn.
- `../../scripts/generate_thesis_v3_figures.py`: script tái tạo hình.

## Tạo hình và biên dịch

Từ thư mục gốc repository:

```text
python scripts/generate_thesis_v3_figures.py
```

Từ thư mục `reports/thesis_v3`, biên dịch bằng XeLaTeX hoặc Latexmk với XeLaTeX. Cần chạy
đủ số vòng để cập nhật mục lục và tham chiếu chéo.

## Nội dung còn phải điền thủ công

- Lời cam đoan và lời cảm ơn.
- Học hàm, học vị của giảng viên hướng dẫn nếu biểu mẫu yêu cầu.
- Tên chính thức của đơn vị đào tạo theo mẫu hành chính hiện hành.
- Kết quả hai thí nghiệm còn thiếu nếu chúng được hoàn thành trước bản cuối.
