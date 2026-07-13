# Speaker notes (Vietnamese)

## Slide 1 — GeoFuse-RNA

Mở đầu ngắn: đề tài xuất phát trực tiếp từ Stanford RNA 3D Folding challenge. Mục tiêu thực dụng vẫn là tạo submission có TM-score cao, nhưng luận văn cần chứng minh một đóng góp nghiên cứu riêng bằng đánh giá có kiểm soát, không chỉ ghép lại notebook leaderboard.

## Slide 2 — Executive summary

Nói kết quả trước. Pipeline temporal-safe đã chạy end-to-end; điểm local mạnh nhất là 0.3072 trên 12 target CASP15. Cải thiện lớn nhất tới từ template recall. Phần refinement v1 cho một negative result hữu ích: sửa đúng metric loss nhưng tạo thêm kink. Vì vậy bước tiếp theo là GeoFuse-RNA, tập trung vào fusion theo từng vùng và geometry v2. Nhấn mạnh private leaderboard chưa có điểm; late submission là phép kiểm chứng bên ngoài sắp tới.

## Slide 3 — Problem and benchmark

Mỗi RNA dài L cần năm dự đoán tọa độ C1′, tức năm tensor L×3. Kaggle chấm best-of-five TM nên một cấu trúc tốt trong năm cấu trúc là đủ. TM chủ yếu nhìn global fold, vì vậy chỉ báo cáo TM có thể che mất clash hoặc backbone phi vật lý. Luận văn báo cáo cả accuracy và validity.

## Slide 4 — Data and EDA

Giải thích ba lớp data: bảng sequence/label; MSA; thư viện PDB_RNA để tìm template. Train v2 tăng từ 844 lên 5.135 sequence và từ 137 nghìn lên 3,68 triệu residue. Validation local có 12 CASP15 target, tổng 2.515 residue, nhưng range 30–720 nt rất lệch nên target dài là stress test thật. PDB library có 8.670 CIF (56,89 GiB); lần parse trước đã cho 23.869 chain và 10,86 triệu residue. Mở `data_audit.md` nếu cần các số đo trực tiếp.

## Slide 5 — Leakage

Đây là slide phương pháp luận quan trọng nhất. CASP15 structure được công bố sau cutoff, nhưng hiện nằm trong PDB dump. Nếu không filter theo ngày, pipeline có thể copy gần đúng native rồi đạt điểm gần 1.0. Ba con số 0.161/0.639/0.957 là cùng pipeline cũ dưới ba chế độ availability, dùng để chứng minh mức inflation. Không dùng oracle score như kết quả thật.

## Slide 6 — Current pipeline

Đi từ sequence tới search, filter, alignment, transfer, gap fill, rồi tạo candidate. Composite search bổ sung remote/weak template mà MMseqs bỏ sót. De novo chỉ là hedge. Cùng module được dùng local và sẽ đóng gói vào Kaggle notebook để tránh “evaluation code khác submission code”.

## Slide 7 — Evolution

Kể như một chuỗi chẩn đoán. Dummy 0.069 là floor. TBM ban đầu 0.161. De novo giúp target không template lên 0.212. Fresh reproduction top-1 đạt 0.2983 và cho thấy search là bottleneck. Khi thêm composite search, score lên 0.3072. Đây là bằng chứng thực nghiệm rằng candidate generation quan trọng hơn optimizer trong regime hiện tại.

## Slide 8 — Candidate recall

MMseqs ban đầu không trả candidate cho 7/12 target temporal-safe. Top-1 luôn tìm một real RNA fold bằng composite similarity, dù không phải homolog mạnh. Real fold đúng kích thước/hình dạng tổng quát vẫn tốt hơn heuristic de novo. Sau khi port ý tưởng này theo temporal-safe rule, pipeline cải thiện 11/12 và hơn reproduced top-1 ở 9/12.

## Slide 9 — Composite ablation

Chỉ ra các case cụ thể: R1117v2 tăng +0.32, R1108 +0.177, R1149 +0.160. Một target giảm nhẹ cho thấy search không phải lúc nào cũng an toàn, nên cần confidence/selection tốt hơn. Runtime khoảng 8 giây mỗi target là chấp nhận được trong notebook 8 giờ.

## Slide 10 — Refinement v1

Không né negative result. Gradient giảm clash 42% và backbone deviation 47%, tốt hơn rule-based trên đúng hai trục này. Nhưng TM giảm 0.002 và sharp kink tăng từ 0.054 lên 0.103. Nguyên nhân hợp lý: loss chỉ ép khoảng cách, nên optimizer có thể bẻ góc để thỏa khoảng cách. Claim đúng hiện tại là “cải thiện distance/clash metrics”, chưa được claim “physical validity nói chung”.

## Slide 11 — Existing contribution

Đóng góp đã có: benchmark temporal-safe, leakage audit, search diagnosis, composite improvement, và adversarial evaluation của refiner. Tuy nhiên top solutions đã dùng TBM + pretrained + refine, nên chỉ gọi generic refinement là novelty sẽ yếu. Đây là lý do plan mới không phủ nhận core Kaggle mà làm integration layer rõ hơn.

## Slide 12 — GeoFuse-RNA

TBM và pretrained tạo fold hypotheses. Mình cluster để không trộn hai fold khác nhau, ước lượng reliability theo residue/segment, fuse các vùng đáng tin, rồi project về geometry hợp lý. Cuối cùng chọn năm cấu trúc cân bằng quality/diversity. Novelty nằm ở cách kết hợp nguồn, không claim model nền là mới.

## Slide 13 — Geometry v2

V2 trực tiếp trả lời failure của v1: thêm angle/curvature, motif context, và tối ưu nhiều stage. Lưu ý thuật ngữ: nếu chỉ giữ C1′ thì gọi là C1′ chain dihedral; standard RNA pseudo-torsion cần thêm P cùng C1′/C4′. Success gate định lượng: kink không cao hơn no-refine, đồng thời clash/backbone vẫn tốt và TM được giữ.

## Slide 14 — Fusion

Confidence không chỉ là một số cho toàn model. Template có thể tốt ở stem nhưng gap ở loop; pretrained có thể ngược lại. Fuse theo segment, sau khi align các candidate trong cùng fold cluster. Dùng smoothing/seam penalty để tránh mỗi residue đổi source tạo “Frankenstein structure”. Nếu train gate, phải dùng prediction thật held-out/family-held-out chứ không chỉ synthetic corruption.

## Slide 15 — Experiment ladder

Mỗi bước trả lời một câu hỏi riêng. B1 raw union trước: nếu oracle candidate pool không tăng thì fusion không thể cứu. B2 chứng minh fusion có hơn chọn whole model không. B3 kiểm tra geometry v2. B4 kiểm tra selector. Selection regret nên định nghĩa là `oracle_pool_TM - selected_best5_TM`, không dùng ratio. Báo paired target results vì n=12 nhỏ.

## Slide 16 — Engineering status

Máy mới đã có env `rna-fold`, CUDA thấy RTX 3060 Ti, MMseqs và test pass. Data 61 GiB đã tải; clean rebuild parse 23.869 chain/10,87 triệu residue, zero error trong 4,6 phút. Fresh runs tái tạo dummy 0.0687, top-1 0.2983 và pipeline 0.3072. Kaggle API token hoạt động nhưng chưa có submission. Hai notebook cũ không còn output session qua API nên không đoán version để nộp. WSL memory setting cần `wsl --shutdown` mới có hiệu lực.

## Slide 17 — Compute strategy

Laptop 1650 không phù hợp rebuild 57 GB CIF hoặc chạy pretrained nặng. Dùng laptop cho code, unit test, phân tích cached predictions, vẽ bảng/slide, viết thesis. Máy 3060 Ti dùng build artifact và DRfold2/RibonanzaNet. Kaggle dùng final offline execution và model cần VRAM cao hơn. Candidate cache giúp tách expensive generation khỏi cheap fusion/refinement ablation.

## Slide 18 — Next steps

Thứ tự có gate: data audit → reproduce B0 → Kaggle baseline → pretrained oracle pool → fusion → geometry v2. Không làm learned gate trước khi chứng minh candidate pool có complementary information. Late submission phải gắn đúng kernel slug/version đã chạy thành công và tạo output `submission.csv`.

## Slide 19 — Questions

Không hỏi chung chung “thầy/cô thấy sao”. Xin quyết định cụ thể về framing, holdout, learned-vs-heuristic gate và success criteria. Nếu supervisor ưu tiên thesis contribution hơn competition score, đề xuất claim hai trục: fusion tăng fold accuracy ở target disagreement/gap, geometry v2 giảm artifact mà không làm mất TM.

## Slide 20 — Evidence map

Đây là appendix để trả lời ngay khi supervisor hỏi “con số này lấy ở đâu”. Mọi headline result đều có report per-target trong repo, không chỉ nằm trên slide. `sources.md` tách nguồn ngoài; `research_plan_review.md` ghi rõ đâu là extension mới và đâu là chỉnh sửa phương pháp cần làm trước khi triển khai.
