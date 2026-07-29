# Giải thích toàn bộ thesis cho hội đồng không chuyên

Tài liệu này trình bày luận văn theo thứ tự từ trực giác đến implementation. Mục tiêu
là để một người chưa học RNA structural biology hoặc protein/RNA folding vẫn hiểu:

1. đầu vào và đầu ra là gì;
2. TBM tạo cấu trúc như thế nào;
3. pretrained model bổ sung điều gì;
4. geometry refinement và GeoFuse đang cố giải quyết vấn đề nào;
5. mỗi thí nghiệm chứng minh hoặc bác bỏ giả thuyết gì;
6. phần nào đã hoàn thành và phần nào chưa.

---

# 1. Trả lời ngắn: thesis đã xong chưa?

## Phần nghiên cứu cốt lõi

Các thí nghiệm đã định trước trong GeoFuse phase 1–7 đã hoàn thành:

- temporal-safe TBM;
- pretrained candidate generation;
- candidate-diversity evaluation;
- geometry v2 với metric độc lập;
- real out-of-fold quality estimator trên 100 RNA;
- global-fold clustering;
- selective fusion với abstention;
- statistical analysis và ablation.

Do đó đã có đủ positive result, negative result và limitations để viết luận văn.

## Những việc chưa đồng nghĩa với “nghiên cứu chưa có kết quả”

Ba sản phẩm triển khai vẫn cần làm tiếp:

1. viết manuscript/chapter hoàn chỉnh từ experiment log;
2. làm slide và hình minh hoạ cho buổi bảo vệ;
3. đóng gói nhánh hybrid thành một Kaggle offline notebook và late-submit.

Submission Kaggle đang có, private `0.60175`, là **TBM-only**. Nhánh hybrid
3 TBM + 2 DRfold2 đã được kiểm chứng trên real-OOF targets nhưng chưa phải một
submission Kaggle mới. Hai kết quả này không được trộn với nhau khi trình bày.

---

# 2. Thesis đang giải quyết bài toán gì?

## 2.1 RNA là gì trong phạm vi bài toán này?

Một RNA được cho dưới dạng chuỗi các nucleotide:

```text
A C G U G G A C ...
```

Mỗi chữ chỉ cho biết loại nucleotide, chưa cho biết chuỗi đó gập trong không gian ra
sao. Cùng một chuỗi thẳng khi viết trên giấy có thể tạo stem, loop, helix, junction
và nhiều global fold khác nhau khi ở trong không gian ba chiều.

## 2.2 Đầu vào

Với mỗi target, pipeline nhận:

```text
target_id
RNA sequence dài L nucleotide
temporal cutoff/release information
```

Ở inference thật, pipeline **không có cấu trúc đúng** của target.

## 2.3 Đầu ra

Kaggle yêu cầu năm cấu trúc dự đoán. Mỗi nucleotide được biểu diễn bằng vị trí atom
C1′:

```text
residue i → (x_i, y_i, z_i)
```

Với RNA dài 80 nucleotide:

```text
1 candidate = ma trận [80, 3]
5 candidates = tensor [5, 80, 3]
```

CSV cuối cùng có dạng:

```text
ID,resname,resid,x_1,y_1,z_1,...,x_5,y_5,z_5
RNA_X_1,A,1,...
RNA_X_2,C,2,...
...
```

## 2.4 Tại sao phải dự đoán năm cấu trúc?

RNA folding là bài toán không chắc chắn. Một sequence có thể có nhiều fold hợp lý,
hoặc model có thể không biết fold nào đúng. Best-of-five cho phép pipeline giữ nhiều
giả thuyết:

```text
candidate 1: fold được nhiều template ủng hộ
candidate 2: fold khác từ pretrained model
candidate 3: một template family khác
candidate 4: biến thể geometry
candidate 5: diversity hedge
```

Kaggle lấy candidate có TM-score cao nhất trong năm cho từng target. Vì vậy năm cấu
trúc giống hệt nhau thường kém hữu ích hơn năm cấu trúc vừa tốt vừa đa dạng.

---

# 3. Câu hỏi nghiên cứu của luận văn

Câu hỏi chính:

> Có thể xây dựng một pipeline RNA 3D temporal-safe, kết hợp template và pretrained
> candidates, để đạt global-fold accuracy tốt hay không?

Câu hỏi đóng góp:

> Khi hai nguồn đúng ở những vùng khác nhau, liệu có thể dùng geometry và
> native-blind confidence để nhận diện vùng tốt, ghép chúng, cải thiện local accuracy
> và cuối cùng tăng TM-score hay không?

Hai cấp độ phải tách riêng:

```text
global fold problem:
    cấu trúc tổng thể gập đúng kiểu hay không?

local geometry problem:
    trong một fold đã gần đúng, các nucleotide cục bộ có nằm hợp lý không?
```

TBM và pretrained model chủ yếu tạo **global-fold hypotheses**. Geometry refinement
và GeoFuse cố sửa hoặc chọn **local regions**.

---

# 4. Workflow tổng thể

```text
                         PREPARATION
 ┌───────────────────────────────────────────────────────────────┐
 │ competition data                                              │
 │ PDB RNA structures                                            │
 │ release dates + target cutoffs                                │
 │ frozen DRfold2 checkpoints                                    │
 │                                                               │
 │ parse template DB → estimate geometry priors → freeze splits  │
 └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     PREDICT ONE RNA SEQUENCE
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
        TBM branch                       pretrained branch
 search past structures                 run frozen DRfold2
 align sequence                         20 checkpoint outputs
 transfer C1′                           confidence ranking
 fill unsupported gaps                 retain top 2
 retain top 3
              │                                │
              └───────────────┬────────────────┘
                              ▼
               normalized raw candidate bank
                     [T1,T2,T3,P1,P2]
                              │
                              ▼
                  pairwise self-TM clustering
               “candidate nào cùng global fold?”
                              │
                              ▼
                   local quality estimation
          “ở residue i, TBM hay pretrained đáng tin hơn?”
                              │
                              ▼
              conservative fusion with abstention
          “chỉ ghép nếu evidence đủ mạnh và liên tục”
                              │
                              ▼
                 optional geometry-v2 projection
                              │
                              ▼
                   quality + diversity selection
                              │
                              ▼
                 five C1′ coordinate predictions
                              │
                              ▼
                       submission.csv
```

Native structure không xuất hiện trong inference path. Nó chỉ xuất hiện trong
training/evaluation path:

```text
train native       → học quality estimator
calibration native → chọn model/threshold
validation native  → đánh giá đúng một lần
test/Kaggle native → không được nhìn thấy
```

---

# 5. Ví dụ xuyên suốt

Giả sử có target:

```text
target_id: RNA_X
length: 80
sequence: ACGGUAC...UAC
cutoff: 2025-01-01
```

Ta chưa biết cấu trúc đúng của `RNA_X`.

Sau toàn bộ candidate-generation:

| ID | nguồn | ý nghĩa |
|:--|:--|:--|
| T1 | TBM | template coverage 90%, sequence khá giống |
| T2 | TBM | template family khác, coverage 72% |
| T3 | TBM | template xa hơn nhưng tạo fold khác |
| P1 | DRfold2 | pretrained hypothesis confidence cao nhất |
| P2 | DRfold2 | pretrained hypothesis thứ hai |

Pipeline không nói:

```text
T1 chắc chắn đúng vì nó có confidence 0.78.
```

Nó chỉ nói:

```text
T1 là TBM candidate được đánh giá tốt nhất theo tín hiệu có sẵn.
P1 là DRfold2 candidate được đánh giá tốt nhất trong thang của DRfold2.
Hai confidence scale chưa chắc so sánh trực tiếp được.
```

Đây là lý do cần source balancing, clustering và learned router.

---

# 6. Data preparation

## 6.1 Competition data

Các nhóm file chính:

- sequence files: target ID, sequence, cutoff và metadata;
- label files: một hoặc nhiều native C1′ conformations cho training/validation;
- test sequences: chỉ có sequence, không có native;
- PDB RNA archive và auxiliary sequence/search data.

## 6.2 Template database

Pipeline parse 8,670 `PDB_RNA/*.cif` thành khoảng 23,869 RNA/RNA-DNA chains và hơn
10 triệu residues.

Mỗi template record chứa:

```text
PDB ID
chain ID
canonicalized RNA sequence
C1′ coordinates
resolved/missing mask
release date
```

Modified nucleotide được map về A/C/G/U khi mapping có cơ sở. C1′ bị thiếu được giữ là
NaN ở bước parse, thay vì giả vờ rằng atom đó tồn tại.

## 6.3 Tại sao release date quan trọng?

Giả sử target cần được dự đoán như tại ngày `2025-01-01`.

| template | release | identity | được dùng? |
|:--|:--|--:|:--|
| 7AAA_A | 2022-03-01 | 62% | có |
| 8BBB_B | 2024-06-10 | 48% | có |
| 9XXX_A | 2025-03-02 | 95% | không |

`9XXX_A` rất giống nhưng xuất hiện sau cutoff. Dùng nó giống như xem đáp án của kỳ thi
năm sau rồi giả vờ đã dự đoán được từ trước.

Pipeline dùng strict rule:

```text
template.release_date < target.temporal_cutoff
```

Ngoài ra, PDB ID gắn trực tiếp với target cũng bị loại để chặn self-template leakage.

## 6.4 Geometry priors

Từ training structures cũ hơn cutoff, pipeline học các thống kê chung:

- adjacent C1′ distance;
- radius-of-gyration theo chiều dài;
- pseudo-angle distribution;
- signed pseudo-torsion distribution;
- pair-like và unpaired local contexts.

Ví dụ prior v1:

```text
mean adjacent C1′ distance ≈ 6.09 Å
clash proxy minimum ≈ 4.18 Å
Rg ≈ 5.18 × L^0.346
```

Geometry v2 dùng 3,397 chains temporal-safe để ước lượng hơn hai triệu local angles
và torsions.

Các prior này trả lời:

```text
“RNA trace nói chung thường trông như thế nào?”
```

chứ không trả lời:

```text
“native của RNA_X nằm ở đâu?”
```

---

# 7. TBM branch — Template-Based Modeling

## 7.1 Trực giác

TBM dựa trên nhận xét:

> Hai RNA có sequence tương tự thường có khả năng chia sẻ một phần structural motif
> hoặc global fold.

Ví dụ đời thường: nếu cần dựng một ngôi nhà mới nhưng có bản thiết kế của một ngôi nhà
rất giống, ta sao chép phần tương ứng rồi sửa phần phòng bị thêm/bớt.

TBM không phải copy nguyên template một cách mù quáng. Nó gồm:

```text
search → temporal filtering → alignment → coordinate transfer
→ gap reconstruction → confidence → refinement → candidate selection
```

## 7.2 Hai tầng template search

### Tầng 1: MMseqs2

MMseqs2 là search engine nhanh. Nó tìm các template có sequence match đủ rõ.

Ưu điểm:

- nhanh;
- phù hợp exact/near homolog;
- scale tốt với template database lớn.

Nhược điểm trong RNA:

- remote hoặc partial similarity có thể bị k-mer prefilter bỏ lỡ;
- một số no-homolog targets trả về zero hit.

### Tầng 2: composite similarity

Pipeline thêm exhaustive/high-recall fallback dựa trên:

- global sequence similarity;
- local Smith–Waterman similarity;
- RNA k-mer/sequence-feature similarity.

Hit từ MMseqs2 và composite search được merge, temporal-filter và re-rank.

Ablation:

```text
MMseqs-only temporal-safe best-of-five TM: 0.2117
MMseqs + composite fallback:             0.3072
delta:                                  +0.0955
```

Điều này cho thấy cải thiện lớn đến từ **template recall**, không phải chỉ từ một
optimizer phức tạp hơn.

## 7.3 Sequence alignment

Sau khi tìm template, target và template được global-align:

```text
target:   A C G G U A C - - G U
template: A C G A U A C U A G U
```

Scoring implementation:

```text
match:        +2
mismatch:     -1
gap open:     -6
gap extend:   -0.5
terminal gap: không phạt
```

Không phạt terminal gap cho phép một template chỉ cover một phần RNA.

## 7.4 Coordinate transfer

Ở aligned target residue có template C1′ hợp lệ:

```text
target_coords[i] = template_coords[j]
support_mask[i] = true
```

Ở insertion, missing template atom hoặc vùng không align:

```text
target_coords[i] = missing
support_mask[i] = false
```

`support_mask` là một distinction quan trọng:

```text
coordinate tồn tại sau gap-fill ≠ coordinate có template evidence
```

## 7.5 Gap reconstruction

Sau transfer, output có thể giống:

```text
known known known [gap gap gap] known known [terminal gap]
```

Pipeline xử lý:

- short internal gap: nội suy giữa hai đầu;
- long internal gap: nội suy cộng curvature vuông góc để tránh một thanh thẳng;
- terminal gap: kéo dài theo backbone direction gần nhất;
- không có điểm nào: extended-chain fallback/de-novo hedge.

Filled residues có confidence thấp hơn. Chúng là giả thuyết geometry để tạo một trace
liên tục, không phải template evidence.

## 7.6 Template confidence

TBM candidate confidence:

```text
confidence = identity × coverage × completeness
```

Trong đó:

- identity: phần trăm aligned positions giống nucleotide;
- coverage: phần trăm target nhận được coordinate thật từ template;
- completeness: phần trăm template chain có C1′ resolved;
- temporal validity: hard gate, invalid candidate bị loại.

Ví dụ:

```text
identity     = 0.70
coverage     = 0.80
completeness = 0.95

confidence = 0.70 × 0.80 × 0.95 = 0.532
```

Đây là engineering confidence, không phải xác suất `53.2% structure đúng`.

## 7.7 TBM candidate diversity

Pipeline giữ template từ distinct PDB/families thay vì năm copy gần giống nhau.

Ví dụ:

```text
T1: strong homolog, cluster A
T2: weaker template, cluster B
T3: partial template, cluster C
```

T3 có thể confidence thấp hơn nhưng hữu ích như một alternative fold.

## 7.8 Geometry refinement v1 trên TBM

Sau gap fill, coordinates được tối ưu bằng energy:

```text
E(X) =
  template_anchor
  + backbone_spacing
  + clash_penalty
  + radius_of_gyration
```

Ý nghĩa:

- giữ phần template-supported gần vị trí gốc;
- cho phép gap/low-confidence regions di chuyển nhiều hơn;
- tránh adjacent C1′ quá xa hoặc quá gần;
- tránh trace collapse;
- giữ kích thước tổng thể hợp lý theo sequence length.

Adaptive strength:

```text
s = 0.2 + 0.8 × (1 − template_confidence)
```

Template mạnh được sửa nhẹ. Template yếu/gap nhiều được sửa mạnh hơn.

## 7.9 TBM-only Kaggle result

Submission hiện có:

```text
public:  0.60084
private: 0.60175
```

Đây là bằng chứng pipeline TBM hoạt động tốt trong benchmark. Nó chưa chứa GeoFuse
pretrained candidate branch.

---

# 8. Pretrained branch — DRfold2

## 8.1 Pretrained nghĩa là gì?

DRfold2 là model đã được tác giả khác train trên RNA structural data. Thesis:

- không train một foundation model từ đầu;
- dùng frozen checkpoints làm candidate generator;
- không sửa model weights bằng validation targets;
- tập trung đóng góp vào auditing, candidate integration, quality estimation và
  geometry-aware fusion.

Từ góc nhìn pipeline:

```text
RNA FASTA
→ frozen DRfold2 cfg97 checkpoints
→ model outputs/confidence/pairwise priors
→ Arena coordinate construction/relaxation
→ PDB candidates
```

## 8.2 Tại sao chạy 20 checkpoints?

Different checkpoints tạo các hypotheses hơi khác nhau. Với `RNA_X`:

```text
checkpoint 1  → fold A, mean confidence 0.38
checkpoint 10 → fold B, mean confidence 0.43
checkpoint 16 → fold A', mean confidence 0.44
...
```

Pipeline rank 20 outputs theo mean per-residue model confidence và giữ hai PDB hợp lệ
có confidence cao nhất:

```text
P1, P2
```

Nếu Arena conversion của output đầu thất bại, runner thử output confidence kế tiếp.

## 8.3 Pretrained confidence không phải native accuracy

Model confidence có thể tương quan với quality khi gộp nhiều targets, nhưng không chắc
rank đúng năm candidates của cùng một target.

Trong development experiment:

```text
pooled confidence–TM Spearman ≈ 0.486
mean within-target Spearman ≈ -0.036
```

Do đó không được đơn giản so:

```text
TBM confidence 0.70 > DRfold confidence 0.50
→ chọn TBM
```

Hai thang đo được tạo từ hai cơ chế khác nhau.

## 8.4 Pretrained temporal/overlap audit

Để đánh giá công bằng, real-OOF experiment chỉ dùng targets xuất hiện sau structural
training cutoff `2023-12-31` của frozen DRfold2 setup.

Mục tiêu là tránh:

```text
model đã thấy structure lúc pretraining
→ đánh giá lại trên chính structure đó
→ tưởng là generalization
```

Đây không phải proof tuyệt đối về toàn bộ training provenance của external model,
nhưng là boundary audit tốt nhất có thể thực hiện với thông tin/checkpoint hiện có.

---

# 9. Candidate bank chung

## 9.1 Tại sao cần common contract?

TBM sinh CIF/coordinates/mask. DRfold2 sinh PDB/confidence/pair priors. Nếu downstream
code xử lý từng format riêng, experiment khó kiểm soát.

Mọi candidate được chuẩn hoá thành:

```text
target_id
sequence
candidate_id
kind: template/pretrained/fused
source
model
coords[L,3]
confidence[L]
support_mask[L]
global_confidence
provenance metadata
optional pairwise priors
```

Với `RNA_X`:

```text
raw bank = [T1, T2, T3, P1, P2]
```

## 9.2 Source-balanced selection

Khi confidence chưa calibrated, pipeline round-robin giữa sources. Mục tiêu là tránh:

```text
scale TBM numerically lớn hơn
→ cả năm slots đều bị TBM chiếm
→ pretrained diversity biến mất
```

Đây là native-blind rule: selection không đọc TM/native.

## 9.3 Candidate diversity result

Development set:

```text
TBM oracle TM:                0.3143
TBM + pretrained selected:    0.4713
TBM + pretrained oracle:      0.5123
```

Khi bỏ target overlap-sensitive:

```text
TBM oracle:                   0.3188
union selected:               0.4245
union oracle:                 0.4693
```

Diễn giải:

- pretrained branch thật sự thêm fold TBM không có;
- selected < oracle cho thấy candidate ranking vẫn chưa hoàn hảo;
- bottleneck chuyển từ “tạo fold” sang “chọn fold/source”.

`oracle` ở đây chỉ là analysis:

```text
nếu sau khi thấy native ta được quyền chọn candidate tốt nhất, ceiling là bao nhiêu?
```

Oracle không deploy được.

---

# 10. Global-fold clustering

## 10.1 Vì sao phải cluster trước khi fusion?

Giả sử:

```text
T1 = hình chữ L
P1 = hình chữ L nhưng loop khác
P2 = hình chữ U
```

Ghép một nửa T1 với một nửa P2 có thể tạo một cấu trúc không thuộc fold nào.

Do đó chỉ cân nhắc local fusion khi hai parents đã giống nhau về global fold.

## 10.2 Self-TM

Self-TM so candidate với candidate, không cần native:

```text
selfTM(T1,P1) cao → có thể cùng fold
selfTM(T1,P2) thấp → khác fold
```

Vì absolute coordinate frame không quan trọng, US-align/TM alignment cho phép
rotation/translation trước khi so hình dạng.

## 10.3 Complete-link clustering

Pipeline tạo similarity matrix rồi complete-link cluster.

Complete-link yêu cầu mọi thành viên trong một cluster tương đối tương thích, bảo thủ
hơn việc chỉ cần một cầu nối giữa hai candidates.

Thresholds được thử trên calibration:

```text
0.35, 0.45, 0.55
```

Cả ba cho aggregate giống nhau; `0.35` được freeze theo deterministic ordering.

Với ví dụ:

```text
cluster A: T1, P1
cluster B: T2, P2
cluster C: T3
```

Chỉ `T1/P1` và `T2/P2` là mixed-source pairs hợp lệ để cân nhắc fusion.

---

# 11. “Làm sao biết vùng nào tốt nếu không có native?”

Đây là câu hỏi phản biện quan trọng nhất.

Câu trả lời không phải:

```text
pipeline biết chắc vùng đó tốt.
```

Câu trả lời đúng:

```text
pipeline học một xác suất từ những tín hiệu có ở inference,
được đánh giá out-of-fold trên targets chưa dùng để train.
```

## 11.1 Native-blind features

Với mỗi pair template/pretrained và mỗi residue, router nhìn:

### Confidence features

- template confidence;
- pretrained confidence;
- rank của confidence trong từng source.

Rank source-local giúp giảm vấn đề hai source dùng hai confidence scale khác nhau.

### Template evidence

- residue có coordinate transfer trực tiếp hay chỉ gap-fill;
- khoảng cách tới vùng template-supported.

### Pair disagreement

Pretrained candidate được robust-align vào template. Router đo khoảng cách giữa hai
source tại residue đó.

```text
disagreement nhỏ:
    hai source gần như đồng ý

disagreement vừa:
    có thể có một local region cần chọn

disagreement quá lớn:
    có thể hai fold không tương thích hoặc alignment không an toàn
```

### Local geometry features của mỗi source

- adjacent backbone deviation;
- angle negative log-likelihood;
- torsion negative log-likelihood;
- sharp-kink indicator;
- candidate-derived pair-like indicator.

### Sequence/context features

- nucleotide one-hot A/C/G/U;
- log sequence length.

Không feature nào chứa native coordinates hoặc native TM-score.

## 11.2 Pair-like có phải annotation sinh học thật không?

Không. Pipeline đánh dấu `pair_like` khi:

- hai bases thuộc AU/UA/GC/CG/GU/UG;
- cách nhau đủ xa trên sequence;
- candidate C1′ distance gần khoảng pair-like.

Đây là structural proxy lấy từ candidate, không phải ground-truth base pair.

## 11.3 Quality labels

Primary label là per-residue C1′-lDDT advantage:

```text
pretrained local lDDT − template local lDDT
```

Nếu dương, pretrained tốt hơn ở vùng đó.

Hai ablations:

- point error sau global alignment;
- sliding-window RMSD 15 residues.

Audit cho thấy ba label chỉ agreement khoảng 70%, nên chúng liên quan nhưng không đồng
nghĩa.

## 11.4 Conv1D router

Model nhỏ:

```text
feature sequence
→ Conv1D kernel 5, 32 channels
→ SiLU + dropout
→ Conv1D kernel 5
→ SiLU + dropout
→ 1-channel output
→ sigmoid
```

Output:

```text
p_i = estimated probability pretrained tốt hơn template ở residue i
```

Conv1D nhìn local neighbourhood thay vì xem từng residue độc lập. Ví dụ một loop dài
10 residues có pattern liên tục sẽ hợp lý hơn một quyết định nhảy source ở đúng một
residue.

## 11.5 Real out-of-fold design

100 targets được chia:

```text
60 train
20 calibration
20 newest validation
```

- family/near-duplicate không đi qua split;
- split có thứ tự thời gian;
- model/threshold chọn trên calibration;
- validation chỉ dùng để báo cáo cuối.

So sánh:

- always TBM;
- always pretrained;
- template-gap rule;
- raw-confidence rule;
- logistic regression;
- gradient boosting;
- Conv1D.

## 11.6 Router result

Primary error là `1 − C1′-lDDT`, thấp hơn tốt hơn:

| method | error |
|:--|--:|
| non-deployable residue oracle | 0.193424 |
| **Conv1D** | **0.238994** |
| logistic | 0.247937 |
| gradient boosting | 0.250951 |
| always pretrained | 0.269575 |
| gap rule | 0.322453 |
| raw confidence | 0.326173 |
| always TBM | 0.347735 |

Conv1D:

```text
ROC-AUC: 0.729868
decision accuracy: 0.650573
pretrained chosen: 37.8% residues
```

So với baseline đã freeze từ calibration:

```text
mean improvement: +0.083458
95% target-bootstrap CI: [+0.023419,+0.155562]
improved targets: 14/20
```

Đây là positive contribution:

> inference-available features có đủ signal để chọn locally better source tốt hơn
> các rule đơn giản trên unseen targets.

Nó chưa chứng minh fusion coordinates sẽ tốt.

---

# 12. Fusion methods

## 12.1 Tại sao cần align trước khi ghép?

Hai structures có thể giống hệt nhưng nằm ở hai coordinate frames khác nhau:

```text
P1 = T1 rotated 90 degrees and translated
```

Trung bình coordinates trực tiếp sẽ tạo cấu trúc vô nghĩa. Pipeline robust-superpose
pretrained vào template bằng Kabsch alignment và iterative residual trimming trước.

## 12.2 F0 — raw parents

```text
F0 = [T1,T2,T3,P1,P2]
```

Đây là baseline an toàn.

## 12.3 F1 — heuristic fusion

F1 dùng rule:

- giữ template ở supported/high-confidence regions;
- dùng pretrained nhiều hơn ở template gaps;
- không trung bình nếu supported disagreement quá lớn;
- smooth alpha quanh boundary.

Coordinate:

```text
X_fused(i) = (1 − alpha_i) X_template(i)
             + alpha_i X_pretrained_aligned(i)
```

Hai modes tạo template-conservative và pretrained-heavy hypotheses.

## 12.4 F2 — learned selective fusion

F2 đặt template làm default parent. Nó chỉ chuyển sang pretrained khi:

```text
p_i ≥ calibrated threshold + safety margin
```

và:

1. parents cùng global cluster;
2. local disagreement nằm trong `[0.5 Å,12 Å]`;
3. quyết định kéo dài ít nhất 7 residues liên tục;
4. boundaries được smooth.

Confirmatory values:

```text
router threshold: 0.75
safety margin:    0.15
effective strong evidence: p_i ≥ 0.90
minimum segment:  7 residues
```

## 12.5 Abstention

Nếu không đủ evidence:

```text
return no fused candidate
```

Không phải:

```text
force a low-confidence fusion
```

Raw parents luôn được giữ:

```text
augmented bank = raw parents + safe fused candidates
```

Đây là safety design: một fusion sai không được phép xoá hypothesis gốc.

## 12.6 Ví dụ fusion giả định

Router dự đoán:

```text
residue  1–25: p ≈ 0.40
residue 26–38: p ≈ 0.94
residue 39–80: p ≈ 0.22
```

Nếu `T1/P1` cùng fold và disagreement an toàn:

```text
1–25:  lấy T1
26–38: chuyển dần sang P1
39–80: quay về T1
```

Nhưng nếu high-probability region chỉ dài 3 residues:

```text
abstain
```

## 12.7 F3 — F2 + geometry v2

Nếu F2 tạo candidate, F3 chiếu nhẹ candidate đó về vùng local geometry prior hợp lý.

Trong confirmatory run, F2 không tạo candidate nên F3 cũng không có input mới.

## 12.8 F4 — native-guided diagnostic

F4 đọc native để hỏi:

```text
nếu biết source nào local tốt hơn thì constructed hybrid trông ra sao?
```

F4 không deploy được và không phải mathematical upper bound của TM-score. Nó chỉ là
diagnostic về headroom.

---

# 13. Geometry refinement v2

## 13.1 Khác geometry v1 ở đâu?

Geometry v1 tập trung:

- adjacent distance;
- clash proxy;
- radius of gyration;
- source/template anchor.

Geometry v2 thêm empirical local shape:

- pseudo-angle distribution;
- signed pseudo-torsion distribution;
- pair-like/unpaired contexts;
- robust backbone loss;
- kink barrier;
- confidence-dependent anchor.

## 13.2 Pseudo-angle và torsion là gì?

Với ba C1′ liên tiếp:

```text
C1′(i−1) — C1′(i) — C1′(i+1)
```

ta đo góc tại residue `i`.

Với bốn C1′ liên tiếp, ta đo signed pseudo-torsion, cho biết backbone xoắn sang hướng
nào.

Đây là coarse C1′ trace geometry, không phải full chemical bond geometry.

## 13.3 Geometry energy trực giác

```text
minimize:
  source-anchor loss
  + robust adjacent-distance loss
  + context angle/torsion NLL
  + clash proxy
  + size/Rg regularization
  + kink barrier
```

Nếu source confidence cao, anchor mạnh và structure ít di chuyển. Nếu confidence thấp,
geometry prior có nhiều quyền sửa hơn.

## 13.4 Independent geometry result

Trên cùng 60 candidates:

| metric | raw | geometry v2 |
|:--|--:|--:|
| best-of-five TM | 0.471268 | 0.471308 |
| C1′-lDDT | 0.472117 | 0.481823 |

C1′-lDDT delta:

```text
+0.009706
95% CI [+0.007403,+0.012617]
12/12 targets improved
56/60 candidates improved
```

Sliding-window RMSD:

```text
9 residues:  improve 0.0572 Å
15 residues: improve 0.0236 Å
31 residues: worsen 0.0049 Å, essentially unchanged
```

Diễn giải:

> geometry v2 cải thiện short-range C1′ trace accuracy nhưng không tìm ra một global
> fold mới.

---

# 14. Final quality-diversity selection

Khi bank có nhiều hơn năm candidates, pipeline chấm native-blind quality từ:

- source-local confidence rank;
- support fraction;
- pair-like fraction;
- angle/torsion likelihood;
- clash/backbone/kink diagnostics.

Sau đó greedy selector cân bằng:

```text
quality
+ bonus cho cluster chưa được cover
+ bonus cho cluster có nhiều model ủng hộ
− penalty nếu quá giống candidate đã chọn
```

Ví dụ:

```text
candidate quality cao nhất thuộc cluster A → chọn
candidate thứ hai cũng cluster A và gần như identical → bị redundancy penalty
candidate hơi thấp hơn thuộc cluster B → có thể được chọn để cover fold khác
```

Trong current confirmatory fusion run, F2 abstain nên bank còn đúng năm raw parents.
Không có fused candidate mới để selector cân nhắc.

---

# 15. Evaluation metrics

## 15.1 TM-score

TM-score đo global-fold similarity sau alignment:

- gần 1: global shape rất giống;
- thấp: global fold khác;
- length-normalized;
- ít bị một vài outlier chi phối hơn RMSD.

Đây là endpoint phù hợp với Kaggle.

## 15.2 Global C1′ RMSD

Sau optimal rigid alignment:

```text
RMSD = căn bậc hai của mean squared C1′ displacement
```

Nhược điểm: một domain bị lệch có thể làm RMSD toàn structure rất lớn.

## 15.3 Sliding-window C1′ RMSD

Thay vì fit cả RNA, lấy cửa sổ 9, 15 hoặc 31 residues và fit riêng.

Ví dụ:

```text
RNA dài 80:
window 1 = residues 1–15
window 2 = residues 2–16
...
```

Nó trả lời local segment có đúng shape hay không.

Window sizes là preregistered analysis scales, không phải universal biological law.

## 15.4 C1′-lDDT

Với residue `i`, xét native C1′ neighbours trong bán kính 15 Å. So các pair distances
giữa prediction và native ở thresholds:

```text
0.5 Å,1 Å,2 Å,4 Å
```

Ưu điểm:

- không cần global superposition;
- tập trung bảo toàn local distance relations;
- phù hợp đánh giá local geometry.

Đây là C1′ adaptation của published lDDT, không phải full-atom lDDT.

## 15.5 Custom diagnostics

Các metric sau là project diagnostics:

- sharp-kink fraction;
- C1′ clash proxy;
- adjacent-distance deviation;
- angle/torsion NLL;
- pair-like fraction.

Chúng hữu ích để debug/optimize nhưng không được gọi là official biology metrics.

---

# 16. Confirmatory fusion result

Trên 20 newest validation targets:

| variant | selected TM | selected C1′-lDDT | oracle TM |
|:--|--:|--:|--:|
| F0 raw | 0.588304 | 0.795059 | 0.588304 |
| F1 heuristic | 0.589985 | 0.793841 | 0.592826 |
| F2 learned selective | 0.588304 | 0.795059 | 0.588304 |
| F3 F2 + geometry | 0.588304 | 0.795059 | 0.588304 |
| F4 native-guided diagnostic | 0.589158 | 0.795648 | 0.589160 |

## 16.1 F1 nói gì?

F1 tạo 24 extra candidates trên 12 targets:

```text
selected TM delta: +0.001681
95% CI: [-0.004908,+0.008495]

oracle TM delta: +0.004522
95% CI: [+0.000795,+0.010107]
```

F1 tạo một ít candidate headroom, nhưng native-blind selection chưa khai thác ổn định.
Selected lDDT còn giảm nhẹ.

## 16.2 F2/F3 nói gì?

Không mixed pair nào vượt đồng thời threshold, margin, segment-length và disagreement
conditions. F2 abstain hoàn toàn:

```text
F2 = F0
F3 = F0
20/20 targets tie
0 material regressions
```

Fusion gate fail vì không tạo oracle headroom mới.

Đây là negative result hợp lệ:

> router biết chọn local source tốt hơn ở mức classification, nhưng current decision
> signal chưa đủ mạnh/liên tục để chuyển thành một Cartesian hybrid an toàn.

---

# 17. Toàn bộ thí nghiệm như một chuỗi giả thuyết

| thí nghiệm | giả thuyết | kết quả | kết luận |
|:--|:--|:--|:--|
| temporal leakage demo | leakage có thể làm điểm tăng giả | leaked reproduction ≈ 0.9355 | temporal audit là bắt buộc |
| composite search | high-recall search tìm template MMseqs bỏ lỡ | 0.2117 → 0.3072 | template recall là gain lớn |
| TBM Kaggle submission | TBM pipeline đủ mạnh cho competition | private 0.60175 | main pipeline baseline mạnh |
| pretrained candidate gate | DRfold2 thêm global folds | union oracle 0.5123 vs TBM 0.3143 | pretrained diversity có giá trị |
| geometry v1 | energy sửa local diagnostics | một số metric tốt, kink có thể xấu | cần safer v2 |
| geometry v2 | local accuracy tăng mà TM không giảm | lDDT +0.009706, TM stable | positive local contribution |
| heuristic fusion | rule thủ công ghép source tốt | oracle headroom nhỏ, selection không ổn định | chưa đủ robust |
| synthetic router | model có học được toy corruption không | AUC 0.9869 | implementation hoạt động |
| synthetic→real | toy router transfer không | fusion 0.4462 vs raw oracle 0.6170 | domain gap lớn |
| 5/5/5 real-OOF pilot | real labels sửa domain gap không | AUC 0.4815 | pilot fail, sample quá nhỏ |
| 60/20/20 real-OOF | larger real data + lDDT label generalize không | Conv1D 0.2390, pass | local routing signal tồn tại |
| selective fusion | routing skill tăng structure quality không | abstain, F2=F0 | classification chưa thành fusion gain |

---

# 18. Đóng góp của thesis là gì?

## Contribution 1 — temporal-safe hybrid candidate pipeline

Không chỉ chạy notebook:

- xây template database;
- strict temporal filtering;
- direct PDB exclusion;
- common candidate/provenance contract;
- OOF audit cho pretrained evaluation.

## Contribution 2 — evidence về source complementarity

TBM và DRfold2 thất bại ở các targets khác nhau. Union oracle gain cho thấy việc kết hợp
nguồn có cơ sở thực nghiệm.

## Contribution 3 — geometry-v2 local improvement

Context-aware C1′ angle/torsion refinement tạo small but stable local accuracy gain mà
không làm mất TM-score.

## Contribution 4 — real-OOF local quality estimator

Conv1D router vượt always-source và heuristic baselines trên primary C1′-lDDT.

## Contribution 5 — scientifically honest negative fusion result

Thesis chỉ ra:

```text
local source classification success
≠
Cartesian fusion success
```

Conservative abstention ngăn một learned component chưa đủ chắc phá raw candidate bank.

---

# 19. Điều không được claim

Không nói:

- hybrid pipeline đã đạt Kaggle `0.60175`;
- GeoFuse đã tăng final TM-score;
- geometry v2 sửa base pairing/stacking/sugar pucker;
- C1′ clash proxy là MolProbity clashscore;
- F4 là deployable method;
- DRfold2 confidence là calibrated probability;
- gap-filled TBM residues có template evidence;
- năm candidate oracle là thứ có thể chọn ở test time.

Nói chính xác:

- Kaggle `0.60175` là TBM-only;
- pretrained tăng candidate coverage trong controlled evaluation;
- quality router tăng local source-selection accuracy;
- selective fusion chưa tăng final TM;
- geometry v2 tăng local C1′ trace accuracy.

---

# 20. Cách kể câu chuyện trong 3 phút

> Đầu vào của tôi chỉ là một chuỗi RNA, nhưng đầu ra phải là năm hình dạng 3D. Tôi
> dùng hai nguồn. Nguồn thứ nhất là template-based modeling: tìm RNA cũ có sequence
> giống, kiểm tra template phải xuất hiện trước target, align sequence, chuyển các
> tọa độ C1′ tương ứng và dựng lại các gap bằng prior hình học. Nguồn thứ hai là
> frozen DRfold2, sinh thêm các global-fold hypotheses mà template search có thể bỏ
> lỡ.
>
> Thí nghiệm cho thấy kết hợp candidate sources làm oracle TM tăng mạnh, nghĩa là
> pretrained thực sự bổ sung fold mới. Nhưng khó khăn là không có native structure ở
> test time để biết source nào đúng. Tôi vì vậy xây một Conv1D router chỉ sử dụng
> confidence, template support, cross-source disagreement và local geometry features.
> Trên 100 RNA chia temporal/family-disjoint, router vượt các always-TBM,
> always-pretrained, gap và raw-confidence rules theo C1′-lDDT.
>
> Tôi tiếp tục kiểm tra liệu local routing đó có thể fusion coordinates. Rule
> conservative yêu cầu hai candidates cùng global fold, xác suất ít nhất 0.90 và một
> đoạn liên tục ít nhất bảy residues. Không pair nào vượt đủ tất cả điều kiện, nên
> method abstain và giữ raw parents. Đây là negative global result nhưng là kết luận
> quan trọng: biết source nào local tốt hơn chưa đủ để tạo một hybrid có TM-score tốt
> hơn. Song song, geometry v2 cải thiện C1′-lDDT ổn định mà giữ TM-score, nên
> contribution cuối là candidate diversity, local geometry improvement và một
> audited confidence-aware routing framework với giới hạn fusion được định lượng rõ.

---

# 21. Các câu hội đồng có thể hỏi

## “Có phải chỉ copy top-1 Kaggle không?”

Không. Top solution truyền cảm hứng cho TBM/pretrained candidate strategy. Phần nghiên
cứu riêng gồm:

- temporal/leakage audit;
- composite search ablation;
- standardized multi-source candidate bank;
- geometry v2;
- real-OOF local label design;
- quality-estimator comparison;
- cluster-constrained selective fusion;
- target-level confidence intervals và negative-result analysis.

## “Nếu không có native, làm sao biết đoạn nào tốt?”

Không biết chắc. Model ước lượng từ inference-available features. Độ tin cậy của ước
lượng được kiểm chứng trên held-out targets. Vì nó vẫn có thể sai, fusion dùng margin,
minimum segment và abstention.

## “Tại sao router pass nhưng fusion fail?”

Router được đánh giá trên quyết định:

```text
ở residue này, chọn TBM hay pretrained có local error thấp hơn?
```

Fusion còn phải giải quyết:

- coordinate-frame alignment;
- boundary continuity;
- correlated errors;
- global topology;
- đoạn quyết định phải liên tục;
- averaged coordinates có thể không nằm trên manifold hợp lý.

Đó là bài toán khó hơn classification.

## “Tại sao local lDDT tăng mà TM không tăng?”

TM-score chủ yếu phản ánh global fold. Sửa vài khoảng cách local trong một fold không
đổi topology tổng thể, nên TM có thể gần như đứng yên. Đây chính là finding của
geometry v2.

## “Vì sao chỉ dùng C1′?”

Competition output và labels chính được biểu diễn ở C1′ resolution. Điều này cho phép
đánh giá global/local trace nhưng giới hạn claim về full atomic chemistry.

## “100 targets có đủ không?”

Nó lớn hơn pilot 15 target và có 20 independent final targets, nhưng vẫn là modest
confirmatory set. Vì vậy thesis báo target-bootstrap CI, improved/regressed counts và
không dùng hàng chục nghìn residues như independent samples.

## “Tại sao không relax threshold để fusion xảy ra?”

Sau khi thấy validation, relax threshold rồi báo cùng validation là positive result sẽ
là post-hoc tuning. Có thể làm như exploratory follow-up, nhưng muốn claim
confirmatory cần calibration/new holdout khác.

## “Current final inference nên dùng gì?”

Theo evidence hiện tại:

```text
giữ 3 temporal-safe TBM + 2 frozen DRfold2 raw parents
ưu tiên source/fold diversity
không force F2 fusion
geometry v2 chỉ dùng ở phạm vi đã kiểm chứng và luôn giữ raw parent
```

Sau đó cần đóng gói đúng logic này thành Kaggle offline notebook và late-submit.

---

# 22. Final thesis statement

> Luận văn xây dựng một pipeline dự đoán RNA 3D temporal-safe kết hợp
> template-based modeling và frozen pretrained candidates. Composite template search
> cải thiện global-fold coverage, trong khi pretrained candidates bổ sung các fold mà
> TBM bỏ lỡ. Một context-aware geometry projection cải thiện C1′ local distance
> accuracy mà bảo toàn TM-score. Trên real out-of-fold targets, một native-blind Conv1D
> router học được cách chọn locally better source tốt hơn các fixed baselines. Tuy
> nhiên, conservative coordinate fusion chưa chuyển local routing skill thành global
> TM-score gain và đã abstain thay vì làm hỏng raw candidates. Kết quả xác định rõ cả
> khả năng lẫn giới hạn của confidence-aware geometry fusion cho RNA 3D prediction.

---

# 23. Method nằm ở đâu trong code?

| thành phần | implementation chính |
|:--|:--|
| parse/build template database | `scripts/build_template_db.py` |
| MMseqs search | `src/rna3d/template/mmseqs_search.py` |
| composite fallback | `src/rna3d/template/composite_search.py` |
| temporal confidence/ranking | `src/rna3d/template/confidence.py` |
| alignment và coordinate transfer | `src/rna3d/template/align.py` |
| gap reconstruction | `src/rna3d/template/gap_fill.py` |
| original geometry refiner | `src/rna3d/refine/optimizer.py` |
| candidate contract/cache | `src/rna3d/geofuse/candidate.py` |
| pretrained candidate evaluation | `src/rna3d/geofuse/phase_a.py` |
| DRfold2 Kaggle GPU runner | `kaggle/geofuse_real_oof_drfold2_medium/phase_e_drfold2.py` |
| geometry-v2 priors/refinement | `src/rna3d/geofuse/geometry_v2.py` |
| clustering/heuristic fusion/selection | `src/rna3d/geofuse/phase_c.py` |
| Conv1D features và architecture | `src/rna3d/geofuse/phase_d.py` |
| real-OOF labels/provenance | `src/rna3d/geofuse/real_oof.py` |
| estimator comparison | `scripts/train_geofuse_quality_estimators.py` |
| selective fusion/abstention | `src/rna3d/geofuse/selective.py` |
| final confirmatory runner | `scripts/run_geofuse_confirmatory_fusion.py` |
| TM/RMSD/lDDT/window metrics | `src/rna3d/eval/` |

Các bảng số liệu gốc và chronological experiment narrative nằm trong
`reports/thesis_notes/geofuse_experiment_log.md`.
