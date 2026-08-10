# Báo cáo thực hiện `reports/thesis_v1/comment.md`

Ngày chốt vòng thí nghiệm: 2026-08-10.

## 1. Quyết định phạm vi

Đã làm theo đề xuất quan trọng nhất trong nhận xét: **GeoFuse không còn là một phần của
pipeline hay contribution chính**. Pipeline luận văn được chốt thành:

1. time-safe TBM để sinh ba ứng viên;
2. DRfold2 để sinh hai ứng viên bổ sung;
3. Geometry v2 tinh chỉnh bảo thủ từng ứng viên;
4. xuất năm cấu trúc dự đoán.

GeoFuse chỉ còn một phụ lục exploratory ghi lại giả thuyết, kết quả âm và lý do dừng.
Nó không được dùng trong submission Kaggle private 0.61390.

## 2. Protocol xác nhận độc lập

Bộ 100 RNA được tái sử dụng đúng ba vai trò:

| Split | Số RNA | Khoảng ngày | Vai trò |
|---|---:|---|---|
| train | 60 | 2024-01-10 đến 2024-12-04 | ước lượng prior hình học |
| calibration | 20 | 2024-12-11 đến 2025-01-08 | chọn cấu hình đã khai báo trước |
| validation | 20 | 2025-01-15 đến 2025-03-26 | mở đúng một lần để kiểm định |

Audit trong runner kiểm tra fail-fast rằng mỗi target chỉ có một dòng và không có
`sequence_group` hoặc `family_group` nào đi qua hai split. Kết quả audit hiện tại:

- sequence groups crossing splits: 0;
- family groups crossing splits: 0;
- mỗi RNA có đúng 3 TBM + 2 DRfold2, toàn bộ tọa độ hữu hạn.

Prior chỉ được học từ 60 RNA train. Cấu hình `source_1p5` được chọn trên calibration và
ghi vào frozen JSON trước khi đọc nhãn validation. Khi tính chỉ số cục bộ, conformation
native tham chiếu của mỗi candidate được cố định từ candidate thô để refinement không
thể hưởng lợi bằng cách đổi sang conformation native dễ hơn.

## 3. TBM white-box: thành phần nào thực sự có ích?

Thí nghiệm được chạy trên 20 RNA calibration, luôn loại PDB của chính target và chỉ dùng
template có ngày công bố sớm hơn target.

- MMseqs2 chỉ tìm được template cho 8/20 RNA (40%). Khi tìm được homolog rõ, best-of-5
  TM có điều kiện là 0.678316; không được dùng con số này như mean của cả 20 RNA.
- Composite search có candidate cho 20/20 RNA. Composite đơn đạt 0.376060; union
  MMseqs2 + composite đạt 0.389705. Vai trò chính của composite là tăng availability,
  không phải bảo đảm precision cao hơn MMseqs2.
- Công thức composite đầy đủ không thắng rõ `G-only`: 0.376060 so với 0.381177,
  chênh +0.005117 với CI [-0.002178, 0.015004]. Vì vậy các trọng số composite được mô tả
  là heuristic, không claim là tối ưu.
- Thêm coverage vào reranking tăng best-of-5 từ 0.371900 lên 0.387558; CI còn chứa 0.
  Completeness không tạo thay đổi trên cohort đầy đủ này. Ép distinct-PDB tăng diversity
  nhưng không đổi oracle TM.
- Curved gap filling chỉ có bằng chứng rõ ở gap dài 9--20 nucleotide: SW-RMSD9 tốt hơn
  linear filling 0.036778 Å, CI [0.009289, 0.067605]. Không claim hiệu quả cho mọi gap.
- Bỏ time filter làm mean tăng từ 0.376060 lên 0.403022, minh họa mức lạc quan do leakage;
  CI còn rộng nên đây là leakage diagnostic, không phải một method hợp lệ.

## 4. TBM và DRfold2 có bổ sung nhau không?

Trên 20 RNA validation mới nhất, trước Geometry:

| Candidate bank | Mean best available TM |
|---|---:|
| 3 TBM | 0.565183 |
| 2 DRfold2 | 0.502627 |
| union 3 TBM + 2 DRfold2 | 0.588304 |

DRfold2 thắng TBM ở 8/20 target, TBM thắng ở 12/20. Union tăng 0.023121 so với TBM,
CI [0.006444, 0.047519], không làm target nào kém đi vì đây là oracle best-of-bank.
Ở so sánh công bằng fixed-N=2, thay một trong hai TBM bằng một DRfold2 tăng 0.030546,
CI [0.002949, 0.068185]. Ở fixed-N=3, hiệu ứng là +0.016012 nhưng CI
[-0.002072, 0.041466] còn chứa 0.

Kết luận hợp lệ là **hai nguồn có tính bổ sung theo target**. Không kết luận DRfold2 nhìn
chung mạnh hơn TBM, và không nhầm lợi ích do thêm candidate slot với lợi ích do nguồn.

## 5. Geometry v2 trên validation độc lập

### 5.1 Kết quả chính

| Chỉ số | Raw | Full Geometry v2 | Thay đổi tốt lên |
|---|---:|---:|---:|
| best-of-5 TM | 0.588304 | 0.587197 | -0.001107 |
| C1′-lDDT | 0.659836 | 0.666496 | +0.006660 |
| SW-RMSD9 (Å) | 2.651395 | 2.611010 | +0.040386 |
| SW-RMSD15 (Å) | 4.273393 | 4.256136 | +0.017257 |
| backbone deviation (Å) | 1.040270 | 0.785817 | +0.254453 |
| clash/C1′ | 0.065121 | 0.022802 | +0.042319 |

Paired bootstrap lấy mẫu theo RNA:

- lDDT: +0.006660, CI [0.000933, 0.013093], tốt hơn ở 12/20 RNA;
- SW-RMSD9: +0.040386 Å, CI [0.029354, 0.052090], tốt hơn ở 19/20;
- SW-RMSD15: +0.017257 Å, CI [0.000617, 0.031336], tốt hơn ở 15/20;
- TM: -0.001107, CI [-0.002809, 0.000418], tốt hơn 8 và kém hơn 12 RNA;
- C1′ RMSD toàn cấu trúc và SW-RMSD31 chưa có bằng chứng thay đổi.

Vì vậy validation gate **PASS** theo tiêu chí đã khóa: local metrics cải thiện trong khi
TM nằm trong biên bảo toàn 0.005. Claim đúng là Geometry v2 sửa quan hệ C1′ cục bộ mà
gần như giữ nguyên global fold; không phải Geometry v2 tăng TM-score.

### 5.2 Đối chứng đơn giản và cơ chế

Đối chứng chỉ gồm source restraint + backbone smoothing cũng tăng lDDT 0.007581. Full
Geometry không hơn đối chứng này về lDDT: chênh -0.000921, CI
[-0.003118, 0.001327]. Tuy nhiên full Geometry tốt hơn đối chứng thêm 0.030585 Å ở
SW-RMSD9, CI [0.019640, 0.044115], đồng thời đối chứng đơn giản làm xấu kink,
angle-NLL và torsion-NLL còn full method sửa các diagnostic này.

Leave-one-out ablation trên calibration xác nhận:

- bỏ backbone gần như làm mất lDDT gain;
- bỏ angle làm angle-NLL xấu rõ;
- bỏ torsion làm torsion-NLL xấu;
- bỏ kink làm sharp-kink rate xấu;
- bỏ clash/Rg làm clash rate xấu.

Các diagnostic angle/torsion/clash/kink là đại lượng cơ chế vì nằm trong hoặc gần hàm
mục tiêu. Bằng chứng hiệu quả độc lập phải dựa chủ yếu vào TM, C1′-lDDT và sliding-window
RMSD.

### 5.3 Factorial và subgroup

Factorial fixed-N=3 tách source bank khỏi Geometry:

- hybrid 2T+1D so với 3T, Geometry off: +0.016012 TM, CI chứa 0;
- Geometry trên bank 3T: -0.005852 TM, CI chứa 0;
- Geometry trên bank 2T+1D: -0.002604 TM, CI [-0.005473, -0.000278].

Do đó không có bằng chứng Geometry tạo ra Kaggle/TM gain. Mức tăng private Kaggle từ
0.60175 lên 0.61390 thuộc về **toàn pipeline lai**, không được quy cho Geometry.

Phân tầng cho thấy Geometry có ích nhất ở candidate chất lượng ban đầu thấp:
lDDT +0.016261, CI [0.008900, 0.024089]. Nhóm chất lượng cao có delta -0.001788 và CI
chứa 0. Đây là finding hậu kiểm để tạo giả thuyết về một quality gate trong nghiên cứu
sau; không được dùng để sửa method sau khi đã mở validation.

Failure cases cần trình bày minh bạch gồm `9E2Z_F` và `8K7W_A` có TM giảm khoảng 0.009,
còn `9B0S_Et` có lDDT giảm khoảng 0.011. Refinement bảo thủ vẫn có thể làm candidate đã
tốt kém đi, nên không được mô tả là thắng trên mọi target.

## 6. Nội dung thesis đã thay đổi

- Đổi title, abstract, RQ, contribution và conclusion sang ba trục: time-safe TBM,
  TBM--DRfold2 complementarity, Geometry v2.
- Bỏ RQ4 và toàn bộ GeoFuse khỏi methodology/results/discussion chính.
- Viết lại thiết kế thí nghiệm 60/20/20, paired target bootstrap, component ablation,
  dumb baseline và fixed-N factorial.
- Cập nhật kết quả Kaggle: TBM public/private 0.60084/0.60175; pipeline cuối
  3 TBM + 2 DRfold2 + Geometry v2 đạt 0.62809/0.61390.
- Ghi rõ pair-like fraction, kink, clash, Rg, angle/torsion NLL là diagnostic nội bộ,
  không phải metric sinh học chuẩn độc lập.
- GeoFuse chỉ còn Phụ lục A với kết quả âm và quyết định loại khỏi core.

## 7. Những việc chưa được phép claim hoặc chưa đủ dữ liệu

1. Chưa thể chạy sweep fixed-five đầy đủ 5T+0D đến 0T+5D vì cache mỗi RNA chỉ có 3 TBM
   và 2 DRfold2. Không đệm bản sao để giả có đủ candidate.
2. Không có bằng chứng Geometry tăng TM-score hoặc đứng sau mức tăng Kaggle.
3. Không có bằng chứng full Geometry tốt hơn dumb baseline trên mọi metric độc lập.
4. Không được claim hiệu quả cho RNA rất dài, multi-chain hoặc full-atom chemistry;
   validation hiện có 20 RNA dài tối đa 96 nucleotide và biểu diễn C1′.
5. Quality gate cho candidate yếu chỉ là hướng nghiên cứu mới; muốn dùng phải
   preregister và kiểm định trên cohort chưa mở khác.

## 8. Tệp tái lập chính

- Runner confirmatory: `scripts/run_geometry_v2_confirmatory.py`
- TBM component study: `scripts/run_tbm_whitebox_ablation.py`
- Hybrid analysis: `scripts/summarize_hybrid_complementarity.py`
- Frozen configuration: `data/processed/geometry_v2_confirmatory/frozen_geometry_v2_config.json`
- Prior provenance: `data/processed/geometry_v2_confirmatory/prior_provenance.json`
- Chi tiết Geometry: `reports/thesis_notes/geometry_v2_confirmatory.md`
- Chi tiết TBM: `reports/thesis_notes/tbm_whitebox_ablation.md`
- Chi tiết complementarity: `reports/thesis_notes/hybrid_complementarity.md`
- Thesis: `reports/thesis_v1/thesis_v1_vi.tex` và `reports/thesis_v1/thesis_v1_vi.pdf`

Toàn bộ 55 unit tests pass; các runner mới cũng qua `py_compile`. PDF được biên dịch bằng
XeLaTeX thành công, 56 trang, và mục lục chỉ còn GeoFuse trong phụ lục exploratory.
