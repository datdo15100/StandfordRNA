# Bản thảo luận văn song ngữ

Thư mục này chứa bản viết lại hoàn chỉnh theo phạm vi kết thúc ở geometry refinement.
Hai ngôn ngữ dùng cùng cấu trúc, số liệu, hình và danh mục tài liệu tham khảo.

## Tệp chính

- `thesis_vi.tex` và `thesis_vi.pdf`: bản tiếng Việt, 54 trang.
- `thesis_en.tex` và `thesis_en.pdf`: bản tiếng Anh, 52 trang.
- `references.tex`: 28 tài liệu dùng chung, giữ nguyên tiêu đề tiếng Anh.
- `figures/`: chín hình ở cả định dạng PDF vector và PNG.
- `citation_audit.md`: nguồn và trạng thái kiểm tra của từng nhóm trích dẫn.
- `../../scripts/generate_thesis_v2_figures.py`: script tái tạo hình.

## Mạch lập luận

Luận văn được tổ chức theo ba câu hỏi:

1. Tìm kiếm composite bổ sung gì cho MMseqs2 trong TBM an toàn thời gian?
2. DRfold2 có bổ sung các target mà TBM bỏ lỡ khi kiểm soát số candidate không?
3. Geometry refinement có cải thiện trace cục bộ trong khi bảo toàn global fold không?

Giải thích trung tâm cho điểm private 0.61390 là candidate-source complementarity. Bằng
chứng local cho thấy geometry refinement cải thiện C1'-lDDT và sliding-window RMSD nhưng
không làm tăng TM-score. Vì vậy điểm Kaggle được báo ở mức toàn pipeline, còn cơ chế của
từng thành phần được giải thích bằng ablation local.

## Hình minh họa

Tám hình là sơ đồ hoặc đồ thị được tạo trực tiếp từ thiết kế và aggregate frozen. Hình
`structure_overlay_placeholder` là placeholder có chủ ý. Để hoàn thiện, cần chọn một RNA
validation có native công khai, chồng native, raw prediction và refined prediction bằng
cùng một quy ước Kabsch, rồi ghi target ID, source, metric và artifact path trong caption.

## Việc cần điền thủ công

- Lời cam đoan theo mẫu trường.
- Lời cảm ơn.
- Học hàm, học vị của giảng viên hướng dẫn nếu biểu mẫu yêu cầu.
- Hình structure overlay theo protocol trong phụ lục.
- Kiểm tra lại tên chính thức của đơn vị đào tạo trên bìa theo mẫu hành chính mới nhất.

## Biên dịch

Đứng tại thư mục này và chạy:

```text
tectonic thesis_vi.tex
tectonic thesis_en.tex
```

Hai nguồn đã được biên dịch thành công bằng Tectonic 0.15.0. Cảnh báo còn lại chỉ là
underfull box do bảng và đường dẫn dài; không có citation hoặc cross-reference chưa định
nghĩa và không có overfull box trong lần biên dịch cuối.
