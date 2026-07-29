# Toàn bộ flow RNA 3D — từ data đến prediction

Tài liệu này giải thích pipeline bằng một ví dụ giả định, dành cho người chưa làm RNA,
structural biology hoặc machine learning.

## Một câu tóm tắt

Với một chuỗi RNA, pipeline tìm vài RNA đã biết có chuỗi tương tự để mượn hình dạng,
xin thêm vài dự đoán từ model pretrained, chỉ ghép các đoạn khi có đủ bằng chứng, rồi
chọn năm cấu trúc vừa đáng tin vừa khác nhau để nộp Kaggle.

```text
sequence + cutoff + template database + frozen pretrained model
              │
              ├── template search → 3 TBM structures ─────┐
              │                                            │
              └── DRfold2 prediction → 2 model structures ─┤
                                                           ▼
                                               normalized candidate bank
                                                           │
                                    global-fold clustering + local quality estimate
                                                           │
                                     selective fusion/geometry (có thể abstain)
                                                           │
                                      giữ raw parents + chọn quality/diversity
                                                           ▼
                                               5 × [L, 3] C1′ coordinates
                                                           │
                                                           ▼
                                                   submission.csv
```

## Ví dụ xuyên suốt

Giả sử test set có RNA `RNA_X`:

```text
target_id: RNA_X
sequence:  ACGGU...UAC       (80 nucleotide)
temporal_cutoff: 2025-01-01
```

Kaggle yêu cầu năm dự đoán. Mỗi dự đoán có đúng một điểm C1′ `(x, y, z)` cho mỗi
nucleotide, nên output của target này có shape:

```text
5 structures × 80 residues × 3 coordinates
```

Pipeline không cần đoán vị trí tuyệt đối trong phòng. Một cấu trúc quay hoặc tịnh tiến
vẫn là cùng một fold; điều quan trọng là hình dạng và quan hệ tương đối giữa các điểm.

---

## Phần A — chuẩn bị data một lần

### A1. Data cuộc thi

Repo tải và đặt đúng chỗ:

- `train_sequences.v2.csv`: sequence, ngày/cutoff và metadata;
- `train_labels.v2.csv`: native C1′ coordinates của training RNA;
- `validation_sequences.csv` và `validation_labels.csv`;
- `test_sequences.csv`: sequence cần predict, không có native;
- `PDB_RNA/`, sequence database, release dates và MSA.

Native label chỉ được dùng để:

1. học generic geometry priors trên phần dữ liệu được phép;
2. tạo supervision cho train/calibration của quality router;
3. đánh giá experiment sau khi method đã được freeze.

Native không đi vào feature hoặc quyết định khi inference.

### A2. Template database

Pipeline parse các RNA structure trong PDB thành:

```text
template sequence
+ C1′ coordinates
+ release date
+ chain/PDB provenance
```

MMseqs2 tạo index để tìm sequence tương tự nhanh. Release date rất quan trọng: khi
giả lập một target ở thời điểm `T`, chỉ template có trước `T` mới hợp lệ. Nếu cho một
structure tương lai vào search thì điểm có thể rất cao nhưng đó là leakage, không phải
prediction.

### A3. Generic geometry priors

Từ training structures cũ hơn cutoff, pipeline ước lượng phân bố:

- khoảng cách C1′ liên tiếp;
- pseudo-angle;
- signed pseudo-torsion;
- context pair-like/unpaired.

Đây là thống kê chung của RNA trace, không phải đáp án riêng của `RNA_X`.

### A4. Frozen pretrained model

DRfold2 đã được tác giả khác train trước. Thesis không train lại model RNA 3D lớn từ
đầu; pipeline chạy checkpoint đã đóng băng để sinh structure và confidence.

Trong real-OOF experiment, chỉ target sau structural-training cutoff của DRfold2 mới
được dùng. Điều đó ngăn model đã học chính structure cần đánh giá.

---

## Phần B — predict một target

## B1. Đọc sequence và luật thời gian

Với `RNA_X`, pipeline lấy:

```text
sequence = 80 nucleotide
cutoff   = 2025-01-01
```

Không có dòng nào kiểu `load native(RNA_X)` trong inference path.

## B2. Nhánh template-based modeling (TBM)

### Tìm template

MMseqs2 search `RNA_X` trong template sequence database. Ví dụ trả về:

| hit | release date | identity | coverage | hợp lệ? |
|:--|:--|--:|--:|:--|
| `7AAA_A` | 2022-03-01 | 62% | 90% | có |
| `8BBB_B` | 2024-06-10 | 48% | 72% | có |
| `9XXX_A` | 2025-03-02 | 95% | 100% | không, ở tương lai |

Hit tương lai bị loại dù đẹp nhất.

### Align và chuyển tọa độ

Sequence alignment cho biết nucleotide nào của target tương ứng với nucleotide nào
của template:

```text
target:   A C G G U A C - - G U ...
template: A C G A U A C U A G U ...
mapping:  1 2 3 4 5 6 7 gap gap 10 ...
```

Ở residue đã map, pipeline mượn C1′ coordinate từ template. Ở vùng gap, nó dựng bridge
liên tục theo geometry prior. Vì thế gap-filled coordinate là hữu hạn nhưng có
`support_mask = false`: có tọa độ không đồng nghĩa có bằng chứng template trực tiếp.

Ví dụ nhánh này sinh:

```text
T1: template tốt, coverage 90%, confidence 0.78
T2: template khác fold, coverage 72%, confidence 0.54
T3: template xa hơn, coverage 60%, confidence 0.41
```

## B3. Nhánh pretrained

DRfold2 chỉ nhìn sequence (và những input model yêu cầu), chạy 20 checkpoint của cfg97,
rồi xếp output theo model confidence. Pipeline giữ hai candidate:

```text
P1: pretrained candidate, confidence per residue
P2: pretrained candidate khác
```

Confidence này là ước lượng của model, không phải native accuracy. Thang confidence của
DRfold2 cũng không tự động so sánh trực tiếp được với confidence TBM.

## B4. Chuẩn hóa candidate

Mọi nguồn được đổi sang cùng một contract:

```text
target_id
sequence
coords[L, 3]
confidence[L]
support_mask[L]
source/model/provenance
optional priors
```

Từ đây, code chọn/cluster/fuse không cần biết file gốc là PDB, CIF hay DRfold output.

Candidate bank hiện tại:

```text
raw bank = [T1, T2, T3, P1, P2]
```

---

## Phần C — hiểu global fold trước khi ghép local

## C1. Self-TM clustering

Pipeline so các candidate với nhau bằng self-TM, không cần native. Giả sử thu được:

```text
cluster A: T1, P1      (cùng global fold)
cluster B: T2, P2      (một fold khác)
cluster C: T3          (fold riêng)
```

Ý nghĩa:

- `T1` và `P1` đủ giống ở mức global để cân nhắc local fusion;
- không ghép `T1` với `P2` nếu chúng thuộc hai fold khác nhau;
- nhiều cluster giúp năm prediction có diversity.

Threshold 0.35/0.45/0.55 được so trên calibration. Chỉ một threshold được freeze trước
khi mở final validation.

## C2. “Đoạn nào tốt” khi không có native?

Ở test time, không thể biết chắc. Pipeline chỉ có các tín hiệu native-blind:

- TBM residue có support trực tiếp hay là gap-fill;
- confidence và rank confidence trong từng source;
- hai candidate bất đồng bao nhiêu sau robust alignment;
- local backbone/angle/torsion diagnostics;
- nucleotide identity và sequence length.

Real-OOF quality estimator học quan hệ giữa các tín hiệu này và source nào có local
C1′-lDDT tốt hơn. Training dùng native của 60 target; threshold/model selection dùng
20 calibration target; 20 target mới nhất chỉ được đọc một lần để test.

Điều estimator trả ra là:

```text
p_i = xác suất pretrained tốt hơn TBM ở residue i
```

Đó vẫn là estimate, không phải ground truth. Vì vậy pipeline có cơ chế abstention.

---

## Phần D — selective fusion có abstention

Xét pair `T1/P1` trong cùng cluster A. Giả sử router trả:

```text
residue  1–25: p ≈ 0.48   → không chắc
residue 26–38: p ≈ 0.88   → pretrained có vẻ tốt hơn
residue 39–80: p ≈ 0.20   → TBM có vẻ tốt hơn
```

Và `T1` đúng lúc có gap ở 28–35.

Pipeline không lập tức blend mọi residue. Nó kiểm tra:

1. `p` phải vượt calibrated threshold cộng margin;
2. đoạn phải liên tục ít nhất bảy residue;
3. hai source phải cùng global cluster;
4. disagreement không quá nhỏ và không lớn đến mức nguy hiểm.

Nếu cả bốn điều kiện đạt, nó tạo candidate `F1`:

```text
1–25   lấy T1
26–38  lấy P1 đã align
39–80  lấy T1
boundary được smooth nhẹ
```

Nếu vùng 26–38 chỉ dài ba residue hoặc probability 0.58 chưa đủ margin, pipeline
**không ghép**. Đó là abstention.

Quan trọng hơn, tạo `F1` không xóa `T1` hoặc `P1`:

```text
augmented bank = raw parents + fused candidates
```

Fusion sai vì thế không bắt buộc phá toàn bộ bank.

### Kết quả confirmatory thật khác ví dụ minh hoạ thế nào?

Trên 20 target cuối, Conv1D router thực sự dự đoán source cục bộ tốt hơn mọi rule
đơn giản theo C1′-lDDT. Nhưng sau khi áp dụng đồng thời:

```text
cùng global cluster
+ probability ≥ 0.75 ± margin 0.15
+ đoạn liên tục ≥ 7 residue
+ local disagreement trong vùng an toàn
```

không pair nào đủ bằng chứng để F2 tạo coordinate mới. Nói cách khác, router có signal
nhưng signal chưa đủ mạnh và liên tục để vượt safety gate. F2 đã abstain đúng như thiết
kế; F2/F3 giữ nguyên năm raw parents và không làm xấu target nào.

Đây không phải “code không chạy”. Đây là kết quả của giả thuyết: local classification
tốt hơn chưa tự động biến thành Cartesian fusion tốt hơn.

## D2. Geometry v2 projection

Một bản `F1_geometry` có thể được chiếu nhẹ về vùng geometry prior:

- cải thiện short-range C1′ distance pattern;
- giảm một số trace clash/kink diagnostics;
- neo theo source confidence để tránh trôi fold.

Kết quả hiện có cho thấy geometry v2:

- giữ mean best-of-five TM: `0.471268 → 0.471308`;
- tăng C1′-lDDT: `0.472117 → 0.481823`;
- cải thiện window RMSD 9 và 15 residue;
- không cải thiện global/31-residue structure rõ ràng.

Vì vậy geometry là local projection, không phải bộ máy tìm global fold.

---

## Phần E — chọn năm prediction cuối

Giả sử bank sau augmentation là:

```text
T1, T2, T3, P1, P2, F1, F1_geometry, ...
```

Pipeline không đơn giản lấy năm confidence lớn nhất. Nó cân bằng:

- native-blind quality score;
- cluster coverage/global-fold diversity;
- tránh năm candidate gần như giống nhau.

Ví dụ final five:

```text
1. T1              cluster A, template mạnh
2. F1_geometry     cluster A, local hybrid hypothesis
3. T2              cluster B
4. P2              cluster B nhưng source khác
5. T3              cluster C, diversity hedge
```

Đây là ví dụ về khả năng của pipeline, không phải output của confirmatory run. Với rule
đã freeze, confirmatory run chọn F2 nhưng F2 abstain hoàn toàn, nên final bank an toàn
hiện tại vẫn là năm raw parents `T1, T2, T3, P1, P2`. Heuristic F1 từng tạo thêm
candidate và tăng mean selected TM rất nhỏ `0.588304 → 0.589985`, nhưng làm
C1′-lDDT giảm `0.795059 → 0.793841`; nó không được coi là final win.

## E2. Ghi submission

Với mỗi residue `RNA_X_i`, CSV có:

```text
ID, resname, resid,
x_1, y_1, z_1,
...
x_5, y_5, z_5
```

Validator kiểm tra:

- đủ và đúng thứ tự ID;
- đúng sequence/residue count;
- không NaN;
- đúng năm bộ tọa độ.

Notebook Kaggle offline cuối cùng phải tạo `submission.csv` theo format này.

---

## Hai flow phải tách tuyệt đối

| nghiên cứu/training | inference/Kaggle test |
|:--|:--|
| có native cho train labels | không có native |
| train quality estimator trên 60 target | load frozen estimator |
| chọn model/threshold trên 20 calibration | không tune threshold |
| đánh giá một lần trên 20 validation | chỉ dùng native-blind features |
| tính TM/RMSD/lDDT để kết luận | xuất 5 structures |

Nếu native coordinate đi vào template selection, clustering, fusion probability hoặc
final-five selection của target đang test, đó là leakage.

---

## Mỗi experiment đang trả lời câu gì?

| experiment | câu hỏi |
|:--|:--|
| temporal-safe TBM | tìm được global fold tốt chỉ bằng structure quá khứ không? |
| composite template search | thêm nguồn template có tăng fold coverage không? |
| pretrained candidate bank | pretrained có tìm được fold TBM bỏ lỡ không? |
| geometry v1/v2 | sửa local trace có tốt hơn mà không phá global fold không? |
| Phase C heuristic fusion | rule thủ công có ghép đúng đoạn không? |
| synthetic gate | kiến trúc router có học được một bài toán giả lập không? |
| real-OOF pilot | gate giả lập có transfer sang lỗi thật không? |
| 60/20/20 quality benchmark | thêm real data và label lDDT có làm router generalize không? |
| clustering ablation | threshold nào cho phép ghép trong cùng fold hợp lý? |
| selective fusion F0–F4 | quality gate + abstention có tạo candidate tốt thật không? |

Một kết quả fail vẫn có ý nghĩa: nó loại một cơ chế không generalize và ngăn pipeline
đưa fusion không đáng tin vào submission.

## Câu chuyện thesis ngắn nhất

> Template và pretrained model giải quyết bài toán tìm global fold. Thesis kiểm tra
> liệu một geometry-aware, confidence-aware layer có sửa đúng các vùng local hay không.
> Geometry v2 đã cho một cải thiện local C1′ nhỏ nhưng ổn định mà không mất TM-score.
> Real-OOF Conv1D routing đã vượt các always-source baseline theo C1′-lDDT, chứng tỏ
> native-blind features có signal để chọn source cục bộ. Tuy nhiên selective coordinate
> fusion chưa tạo được candidate mới dưới safety gate; pipeline vì vậy giữ raw parent
> candidates và abstain. Kết quả tách rõ hai bài toán: “biết source nào local tốt hơn”
> và “ghép được một structure có TM-score tốt hơn” không phải cùng một việc.
