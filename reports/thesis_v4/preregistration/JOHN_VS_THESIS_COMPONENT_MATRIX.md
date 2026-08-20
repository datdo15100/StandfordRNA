# John và thesis: ma trận thành phần trước thí nghiệm V4

Trạng thái: audit code lần đầu, chưa phải kết quả reproduction.

Tên baseline bắt buộc trong V4 là **reproduced publicly released John pipeline**. Không dùng cụm “exact winning pipeline” vì code public, data Kaggle gắn kèm, checkpoint và final winning submission chưa được chứng minh là hoàn toàn giống nhau.

| Component | John làm gì | Thesis làm gì | Khác nhau ở đâu |
|---|---|---|---|
| Input/data handling | TBM-only ghép 844 dòng gốc với bảng mở rộng và notebook báo 18.946 sequence rows, 18.815 coordinate groups. Hybrid đọc `merged_sequences_final.csv` và `merged_labels_final.csv`. | Dùng Kaggle train V2, PDB_RNA đã parse và metadata riêng của target. | Hai nguồn input không đồng nhất. Hash của dataset John chưa có nên chưa thể gọi exact. |
| Template database | Public TBM quét bảng sequence/coordinate đã ghép. Bản dựng local John-style hiện chỉ có 7.155 sequence duy nhất. | Database hiện có 23.869 RNA chains thuộc 8.613 PDB, sau đó lọc theo snapshot được phép. | Universe và cách deduplicate khác nhau. J-controlled phải dùng cùng một database đã khóa với thesis. |
| Temporal filtering | Không thấy strict per-target release-date gate trong code public đã lưu. | Chỉ nhận template có `release_date < target cutoff`. | Đây là safeguard của thesis và phải áp dụng cho cả J-controlled. |
| Self/reference exclusion | Không thấy loại trực tiếp PDB của target trong retrieval public. | Loại PDB ID của target lấy từ metadata. | Khi so controlled, cả hai phải dùng cùng self-exclusion. |
| Dedup/clustering | Cluster top candidates bằng feature sequence. KMeans nếu có ít nhất 15 candidates, nếu ít hơn dùng cách chọn xa nhau. | Cohort dùng exact-sequence hash và MMseqs sequence-similarity cluster. Bank ưu tiên distinct PDB. | John cluster candidate để tạo diversity; thesis còn cluster data để chống overlap. Đây là hai việc khác nhau. |
| TBM retrieval | Exhaustive scan sau length filter với `0.4 global + 0.3 local + 0.2 feature + 0.1 Jaccard 3-mer`. | MMseqs2 prefilter cộng exhaustive composite search kế thừa từ John. | Thesis thêm nhánh tìm homolog nhanh nhưng vẫn giữ composite để tăng recall. |
| Ranking | Composite score, sau đó lấy candidate tốt nhất trong từng cluster. | Realign rồi xếp theo `identity × query coverage × template completeness`, ưu tiên PDB khác nhau. | Score và diversity rule khác nhau, phải ablate từng term. |
| Coordinate transfer | `pairwise2` global alignment, copy C1′ từ template. | `PairwiseAligner` global, chỉ copy C1′ hữu hạn. | Alignment score và end-gap policy khác nhau. |
| Gap handling | Linear hoặc thêm một cung sin khi hai anchor quá gần; terminal extension; de novo nếu thiếu toàn bộ. | Curved gap từ gap dài ít nhất 3, terminal extension, gán confidence thấp cho residue được điền; có linear baseline. | Trigger, đường cong và representation của confidence khác nhau. |
| Pretrained model | Hybrid public chạy Boltz-1, rồi DRfold2 `cfg_97`; Boltz model 0 được đưa vào DRfold2 như AF3-style restraint. | Production hybrid chạy direct DRfold2 `cfg_97` + Arena, lấy tối đa hai output theo mean confidence; không dùng Boltz. | John public là Boltz-constrained DRfold2, thesis là direct DRfold2. Không được mô tả như chỉ thay một model. |
| Candidate allocation | Code public route theo target: nhóm đầu dùng 5 template, nhóm sau dùng 5 DRfold2/Boltz cho tới time limit; lỗi thì fallback template. | Bình thường `3T + 2D`; thiếu D thì bù T. | John route cả target, thesis trộn source trong cùng target. Primary H2 isolate allocation bằng `5T vs 3T+2D` cùng Thesis TBM; John-to-thesis bank comparison chỉ là secondary end-to-end evidence. |
| Confidence | Composite similarity điều khiển độ mạnh của John rule-refiner. Không thấy calibration chung giữa các source. | TBM dùng `identity × coverage × completeness`; DRfold2 dùng pLDDT-derived confidence; Geometry đang dùng trực tiếp. | Confidence khác source chưa được calibration. Bắt buộc so với fixed strength. |
| Refinement | TBM dùng `adaptive_rna_constraints` một lượt. Hybrid dùng DRfold2 Selection, Optimization, Arena và Boltz restraint. | Mọi candidate được chạy Geometry 300 bước với source, backbone, clash, Rg, angle, torsion và kink terms. | Chỉ same-candidate factorial mới tách được tác dụng của refiner. |
| Fallback | TBM thiếu candidate thì sinh de novo. Hybrid lỗi hoặc skip DRfold thì về template; nếu DRfold ít hơn 5 output thì duplicate output cuối. | Thiếu DRfold thì thêm TBM; TBM thiếu thì de novo; Geometry lỗi thì giữ raw. | Fallback khác nhau. Nếu chưa đo chất lượng thì chỉ gọi engineering safeguard. |

CSV dùng cho máy đọc: [`john_vs_thesis_component_matrix.csv`](john_vs_thesis_component_matrix.csv).

## Kết luận audit hiện tại

Ba khác biệt lớn nhất cần thí nghiệm kiểm soát là:

1. John và thesis chưa dùng cùng template universe và leakage rules.
2. John public hybrid route cả target giữa template và Boltz-constrained DRfold2, còn thesis tạo bank `3T + 2D` trong từng target. Vì vậy pretrained attribution phải đến từ controlled `5T vs 3T+2D`, không phải từ hiệu hai complete pipelines.
3. Hai bên có refinement hoàn toàn khác loại, nên không thể trừ hai leaderboard score để suy ra Geometry giúp bao nhiêu.

Chi tiết nào chưa có file, version hoặc hash được giữ là `UNKNOWN` trong preregistration, không được điền bằng suy đoán.
