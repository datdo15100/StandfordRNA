Ừ, t đang review đúng **bản LaTeX mới** ní vừa gửi.  Và sau khi đọc bản này + nhìn nó dưới giả định **bỏ GeoFuse khỏi core thesis, dừng ở Geometry v2**, t thấy hướng đó **sạch hơn và mạnh hơn đáng kể**.

Điểm quan trọng là: **đừng cố làm thesis rộng nữa. Hãy làm nó sâu.** Với data và code hiện tại, hoàn toàn có thể biến nó thành một thesis kiểu:

> **“T xây một pipeline RNA 3D prediction time-safe, phân tích định lượng từng nguồn đóng góp ở đâu, rồi đề xuất Geometry v2 và chứng minh bằng ablation + held-out validation rằng refinement này sửa local geometry mà không phá global fold.”**

Cái đó **đủ thesis**, thậm chí thuyết phục hơn câu chuyện GeoFuse hiện tại.

---

# 1. Bản mới này tốt hơn bản trước ở đâu?

Cải thiện quan trọng nhất là giờ ní đã có **external result cho hybrid configuration**:

| System                          |      Public |      Private |
| ------------------------------- | ----------: | -----------: |
| TBM                             |     0.60084 |      0.60175 |
| 3 TBM + 2 DRfold2 + Geometry v2 | **0.62809** |  **0.61390** |
| Δ                               |    +0.02725 | **+0.01215** |



Cái này rất đáng giá.

Ở bản trước, reviewer có thể bảo:

> “Full hybrid của cậu đâu? Tất cả bằng chứng hybrid chỉ có trên 12 CASP15.”

Bây giờ không nói vậy được nữa. Ní đã có **một frozen pipeline chạy end-to-end trên hidden data**.

Nhưng draft cũng tự thừa nhận đúng một vấn đề: submission này thay đổi candidate composition và áp dụng Geometry v2, nên **+0.01215 chưa phân rã được thành contribution riêng của DRfold2 và Geometry v2**.

Đấy chính là nơi thesis V2 nên đào.

---

# 2. Nếu bỏ GeoFuse, t sẽ đổi “identity” của thesis

Title hiện tại vẫn có:

> “… với tinh chỉnh hình học **và ước lượng chất lượng cục bộ**”



Nếu bỏ GeoFuse thì vế cuối phải bỏ luôn.

T sẽ đưa title về kiểu:

**Tiếng Việt**

**Mô hình lai dựa trên khuôn mẫu và học sâu cho dự đoán cấu trúc ba chiều RNA với tinh chỉnh hình học có kiểm soát**

hoặc nhấn vào leakage:

**Dự đoán cấu trúc ba chiều RNA bằng mô hình lai khuôn mẫu–học sâu với kiểm soát rò rỉ thời gian và tinh chỉnh hình học**

**English**

**A Hybrid Template-Based and Deep Learning Framework for RNA 3D Structure Prediction with Conservative Geometric Refinement**

Thesis khi đó có **3 tầng scientific question** rất đẹp:

[
\boxed{\text{Can we retrieve better templates?}}
]

↓

[
\boxed{\text{Do deep models add complementary folds?}}
]

↓

[
\boxed{\text{Can we refine local geometry without destroying the fold?}}
]

Dừng ở đó.

Không cần tầng thứ tư:

[
\text{Can we splice pieces from different structures?}
]

vì tầng đó hiện vừa loãng narrative, vừa không có positive result.

---

# 3. Nếu t là supervisor, đây mới là 3 contribution t muốn ní defend

Hiện contribution section nói **hai methodological contributions + một empirical contribution**, trong đó GeoFuse được đưa lên ngang Geometry.

T sẽ đổi hierarchy thành:

### Contribution A — Time-safe TBM pipeline + component analysis

Không nhất thiết claim thuật toán TBM mới hoàn toàn.

Contribution nằm ở:

> xây dựng, tái lập và **phân rã định lượng** pipeline TBM dưới temporal leakage control.

Cái này rất quan trọng, vì case R1107 đã cho ní một ví dụ cực mạnh: template leak đưa TM lên **0.9355**, trong khi time-safe chỉ **0.2983**.

Đây không còn là “engineering housekeeping”. Nó là **experimental methodology contribution**.

---

### Contribution B — Empirical analysis of TBM–DRfold2 complementarity

Không claim DRfold2 là của mình.

Claim của mình là:

> dưới cùng một protocol time-controlled, deep-learning candidates bổ sung fold hypotheses mà TBM không tìm thấy.

Evidence hiện đã khá đẹp:

* TBM oracle: 0.3143
* hybrid oracle: 0.5123
* loại exact-overlap R1128 vẫn 0.3188 → 0.4693.
* Final hybrid cũng thắng TBM trên Kaggle hidden set.

DRfold2 bản thân nó là một deep RNA predictor có cả denoising structure module và post-processing/model-selection pipeline; paper của DRfold2 cũng báo cáo complementarity với một nguồn prediction khác, nên việc nghiên cứu complementarity như một *system-level question* là hoàn toàn hợp lý. ([PLOS][1])

---

### Contribution C — Geometry v2

**Đây mới là methodological contribution chính.**

Geometry v2 hiện được định nghĩa rất rõ theo triết lý:

[
\text{better local geometry}
\quad\text{subject to}\quad
\text{stay close to }X_0
]

với source restraint + backbone + clash + (R_g) + angle + torsion + kink guard và adaptive scale.

Nếu thesis dừng đây thì t muốn **50% sức nặng scientific experiments nằm ở Geometry v2**.

Không phải một bảng before/after.

Mà là **mổ bụng Geometry v2 ra**.

---

# 4. Cái t muốn thấy nhất: “component → hypothesis → ablation → conclusion”

Đây chính là ví dụ ní nói: “từng contribution của component trong TBM hay Geometry refinement”.

T sẽ tổ chức experiments kiểu này.

## TBM: đừng chỉ hỏi “pipeline có tốt không”

Hiện RQ1 chỉ có:

[
MMseqs2
\quad vs \quad
MMseqs2 + composite\ search
]

và result 0.2117 → 0.3072.

Tốt, nhưng reviewer sẽ hỏi tiếp:

> “**Why?** Component nào tạo ra improvement?”

T muốn một ablation table như:

| Variant                  | Search recall | Top-1 TM | Best-of-5 TM | Coverage |
| ------------------------ | ------------: | -------: | -----------: | -------: |
| MMseqs2 only             |               |          |              |          |
| Composite only           |               |          |              |          |
| MMseqs2 + composite      |               |          |              |          |
| + temporal filtering     |               |          |              |          |
| + reranking              |               |          |              |          |
| + distinct-PDB selection |               |          |              |          |
| + gap filling            |               |          |              |          |
| Full TBM                 |               |          |              |          |

Không nhất thiết mọi row đều phải tăng TM.

Thesis hay chính là khi mình nói được:

> Composite search chủ yếu tăng **recall**.

> Reranking biến recall thành **top-k precision**.

> Distinct-PDB selection không nhất thiết tăng top-1 nhưng tăng **candidate diversity / best-of-5**.

> Gap filling chủ yếu giúp những targets có **low template coverage**.

Đó là scientific understanding.

---

# 5. Composite search cũng nên ablate riêng

Ní có:

[
C=0.4G+0.3L+0.2F+0.1K_3
]

trong đó global alignment, local alignment, sequence composition/features và 3-mer overlap.

Reviewer chắc chắn có thể hỏi:

> “Tại sao 0.4, 0.3, 0.2, 0.1?”

T không cần ní làm hyperparameter search hàng nghìn trial.

Chỉ cần component study:

| Retrieval score     | Best-of-5 TM | Useful-template recall |
| ------------------- | -----------: | ---------------------: |
| (G) only            |              |                        |
| (G+L)               |              |                        |
| (G+L+F)             |              |                        |
| (G+L+F+K_3)         |              |                        |
| Full weighted score |              |                        |

Rồi có một **sensitivity analysis**:

[
w_G \pm 25%,\quad
w_L\pm25%
]

Nếu performance tương đối stable thì reviewer sẽ bớt cảm giác “magic constants”.

Không cần chứng minh 0.4/0.3/0.2/0.1 là global optimum.

Chỉ cần chứng minh:

> conclusion không phụ thuộc vào đúng một bộ weight rất mong manh.

---

# 6. Template ranking cũng là một scientific question riêng

Ní hiện rank bằng:

> identity × target coverage × template coordinate completeness.

Đừng để nó trôi qua như implementation detail.

Test:

| Ranking                            | Top-1 | Top-3 oracle | Top-5 oracle |
| ---------------------------------- | ----: | -----------: | -----------: |
| Identity only                      |       |              |              |
| Identity × coverage                |       |              |              |
| Identity × coverage × completeness |       |              |              |
| Full + distinct PDB                |       |              |              |

Câu kết luận có thể rất hay:

> sequence identity alone is insufficient because a highly similar but poorly covered template may transfer less usable geometry than a moderately similar full-coverage template.

Nếu data support.

Đây là kiểu analysis reviewer thích vì nó cho thấy **mình hiểu pipeline chứ không chỉ chạy code**.

---

# 7. Gap filling cũng nên được đánh giá theo đúng regime của nó

Hiện ní có khá nhiều logic:

* interpolation cho short internal gap,
* curved path cho long gap,
* backbone extrapolation cho terminal gap,
* coarse fallback khi không có template.

Không nên lấy toàn bộ RNA average rồi hỏi gap fill có tốt không.

Hãy stratify theo:

[
\text{unsupported fraction}
]

và:

[
\text{gap length}
]

Ví dụ:

| Unsupported region | Linear | Current gap-fill | Δ local RMSD |
| ------------------ | -----: | ---------------: | -----------: |
| 1–3 nt             |        |                  |              |
| 4–8 nt             |        |                  |              |
| 9–20 nt            |        |                  |              |
| >20 nt             |        |                  |              |

Đây là thứ biến một implementation heuristic thành **evidence-backed engineering choice**.

---

# 8. Với hybrid, t sẽ làm một factorial experiment cực đơn giản

Hiện Kaggle comparison là:

[
TBM
]

vs

[
3TBM + 2DRfold2 + Geometry
]

nên bị confounding.

Trên **held-out labeled data**, chạy đúng 4 cấu hình:

| Candidate source  | Geometry OFF | Geometry ON |
| ----------------- | -----------: | ----------: |
| 5 TBM             |            A |           B |
| 3 TBM + 2 DRfold2 |            C |           D |

Từ đó có ngay:

[
B-A = \text{Geometry effect on TBM}
]

[
C-A = \text{candidate-source effect}
]

[
D-C = \text{Geometry effect on hybrid candidates}
]

và:

[
D-A = \text{full-system gain}
]

**Đây là experiment t đánh giá cao nhất sau independent Geometry validation.**

Bởi nó trả lời đúng điểm yếu mà chính draft đang confess:

> external improvement chưa tách được DRfold2 và Geometry v2.

---

# 9. Thậm chí 3 TBM + 2 DRfold2 cũng cần có lý do

Tại sao là:

[
3+2
]

mà không phải:

[
4+1,\quad2+3,\quad1+4?
]

Chạy trên **calibration set**:

| Allocation | TM selected | TM oracle | diversity |
| ---------- | ----------: | --------: | --------: |
| 5T + 0D    |             |           |           |
| 4T + 1D    |             |           |           |
| 3T + 2D    |             |           |           |
| 2T + 3D    |             |           |           |
| 1T + 4D    |             |           |           |
| 0T + 5D    |             |           |           |

Sau đó **freeze 3+2** nếu nó tốt/robust rồi test trên held-out.

Thế là 3+2 từ “t chọn đại” trở thành **design decision empirically justified**.

---

# 10. Geometry v2: đây mới là chỗ t muốn ní đào thật sâu

Current result actually promising:

[
lDDT:\ 0.472117\rightarrow0.481823
]

[
RMSD_9:\ 4.00131\rightarrow3.94415
]

và lDDT tăng ở **56/60 structures**, tất cả **12/12 RNA**; CI cũng không chứa 0.

Đó là nền rất tốt.

Nhưng một reviewer vẫn hỏi:

> “Which part of Geometry v2 is responsible?”

Và đây là bảng t **bắt buộc muốn có**:

| Model                          | lDDT Δ | RMSD-9 Δ | RMSD-15 Δ | TM Δ | New kinks |
| ------------------------------ | -----: | -------: | --------: | ---: | --------: |
| No refinement                  |      — |        — |         — |    — |         — |
| Source restraint only          |        |          |           |      |           |
| + backbone                     |        |          |           |      |           |
| + angle                        |        |          |           |      |           |
| + torsion                      |        |          |           |      |           |
| + clash + (R_g)                |        |          |           |      |           |
| + kink guard                   |        |          |           |      |           |
| Full, fixed strength           |        |          |           |      |           |
| **Full + adaptive confidence** |        |          |           |      |           |

T không nhất thiết thích *cumulative ablation* duy nhất. Có thể thêm **leave-one-out**:

[
Full-\mathcal L_{angle}
]

[
Full-\mathcal L_{torsion}
]

[
Full-\mathcal L_{kink}
]

để biết component nào **necessary**, không chỉ biết thêm component theo thứ tự nào thì tăng.

---

# 11. Có 3 component của Geometry mà t đặc biệt muốn chứng minh

### Source restraint

Đây là linh hồn của “conservative refinement”.

Reviewer muốn thấy trade-off:

[
\lambda_{source}
\downarrow
\Rightarrow
\text{local correction}\uparrow?
]

nhưng có thể:

[
TM\downarrow
]

Vẽ một curve:

**x-axis:** source restraint strength
**y-axis 1:** ΔlDDT
**y-axis 2:** ΔTM

Nếu xuất hiện vùng mà lDDT tăng trong khi TM gần 0 delta → **đấy chính là empirical justification của conservative refinement**.

---

### Adaptive confidence

Ní đang dùng:

[
s=0.2+0.8(1-c_{global})
]

tức cấu trúc confidence thấp bị refine mạnh hơn.

Đây là một idea hoàn toàn đáng ablate:

[
s=1
]

vs

[
s=f(c_{global})
]

Rồi stratify theo initial quality/confidence.

Nếu adaptive version giúp nhiều hơn ở bad/uncertain structures mà không phá good structures thì cực đẹp.

---

### Kink guard

Ní dành weight **20.0** cho kink term — lớn vãi :)))

Reviewer chắc chắn sẽ hỏi.

Test:

**Full Geometry − kink guard**
vs
**Full Geometry + kink guard**

và đo:

* number of newly created (<70^\circ) kinks,
* number of pre-existing kinks worsened,
* lDDT,
* RMSD-9,
* TM.

Nếu kink guard không cải thiện lDDT nhưng giảm catastrophic local artifacts thì vẫn là contribution rất hợp lý.

---

# 12. Có một chỗ trong Geometry description t muốn sửa ngay

Draft nói empirical geometry lấy:

* consecutive distance,
* pseudo-angle,
* pseudo-torsion,
* radius of gyration,
* non-adjacent-neighbor distance,
* pair-like/unpaired indicator.

Nhưng objective được viết chỉ có:

[
L_{source},
L_{backbone},
L_{clash},
L_{Rg},
L_{angle},
L_{torsion},
L_{kink}.
]



**pair-like đâu? non-adjacent neighbor distribution map vào loss nào?**

Hiện đọc method có cảm giác hai thứ được “learned” nhưng rồi không rõ có tham gia refinement hay chỉ dùng ở downstream selector.

Nếu bỏ GeoFuse/selector thì càng phải dọn:

* nếu không dùng trong Geometry v2 → **remove khỏi Geometry section**;
* nếu dùng → ghi công thức chính xác nó đi vào objective thế nào.

Reviewer rất dễ bắt chỗ này.

---

# 13. Geometry cần một “dumb baseline”

Đây là thứ rất quan trọng.

Nếu baseline duy nhất là:

> no refinement

thì reviewer có thể nói:

> “Có thể bất kỳ smoothing nào cũng tăng local lDDT một chút.”

Do đó t muốn ít nhất một baseline rất ngu nhưng fair:

[
L =
\lambda_sL_{source}
+
\lambda_bL_{bond}
]

chỉ sửa consecutive (C1') distances.

Hoặc một Laplacian/local smoothing baseline constrained to stay near source.

Rồi hỏi:

> Geometry v2 có tốt hơn **simple smoothing** không?

Nếu có → angle/torsion empirical priors thực sự có value.

Nếu không → cũng là discovery quan trọng: method có thể simplify.

Nếu ní dựng được full-atom representations thì có thể cân nhắc external refinement baseline như QRNAS; QRNAS được thiết kế specifically cho fine-grained nucleic-acid refinement với force-field restraints. ([PubMed Central (PMC)][2]) Nhưng vì thesis hiện chỉ xử lý (C1'), t **không bắt buộc** QRNAS — apples-to-apples coarse-grained baseline quan trọng hơn.

---

# 14. Geometry nên trả lời “when does it work?”

Đây là thứ sẽ nâng thesis từ **master project tốt** sang **research thesis khá thuyết phục**.

Đừng chỉ average.

Stratify Geometry gain theo:

[
\text{source}\in{TBM,DRfold2}
]

[
\text{initial TM quartile}
]

[
\text{initial lDDT quartile}
]

[
\text{sequence length}
]

[
\text{template support fraction}
]

[
\text{gap fraction}
]

Ví dụ có thể discover:

> Geometry giúp TBM nhiều hơn DRfold2.

hoặc:

> Geometry chủ yếu giúp medium-quality structures, còn rất tệ thì không cứu được vì fold sai.

hoặc:

> Geometry giúp low-confidence models nhiều hơn high-confidence models.

Đấy mới là **scientific insight**.

---

# 15. Và phải có convergence / robustness

Ní dùng Adam:

[
300\ steps,\ lr=0.04
]



T muốn plot:

[
0,25,50,100,200,300
]

steps:

* objective,
* lDDT,
* RMSD-9,
* TM,
* displacement from source.

Nếu 100 step đã plateau thì reviewer sẽ hỏi tại sao 300.

Nếu metric 계속 improve tới 300 thì justified.

Thêm một sensitivity nhỏ:

[
lr\in{0.02,0.04,0.08}
]

và maybe source-weight ±50%.

Không cần grid search.

Mục đích là chứng minh:

> result không phụ thuộc một magic configuration duy nhất.

---

# 16. Independent validation: đây là thay đổi t ưu tiên số 1

Hiện Geometry dùng 60 structures từ **12 CASP15 development RNAs**.

Đây vẫn là điểm yếu lớn nhất.

Trong khi ní đang có:

* 60 RNA training
* 20 calibration
* 20 newest validation,
* separated by time/family.

Bỏ GeoFuse xong, **repurpose luôn**:

[
\boxed{\text{Train 60}}
]

→ estimate empirical geometry priors.

[
\boxed{\text{Calibration 20}}
]

→ chọn weights, LR, step count, kink threshold, adaptive scheme.

[
\boxed{\text{Test 20 newest}}
]

→ **freeze everything and evaluate once.**

Và 12 CASP15 trở thành:

> development + interpretability + case studies.

Đây là thiết kế t sẽ ký duyệt ngay nếu làm supervisor.

---

# 17. Cách evaluation này cũng hợp với chuẩn benchmark RNA hơn

RNA-Puzzles toolkit nhấn mạnh việc benchmark trên **đa dạng cấu trúc**, dùng nhiều loại structural-comparison metrics và decoy/model sets thay vì chỉ một score; toolkit cũng có riêng các metric về RMSD, deformation và interaction fidelity. ([OUP Academic][3])

Thesis của ní chỉ có (C1'), nên không thể dùng hết full-atom metrics — và draft đã acknowledge điều đó. Nhưng principle nên giữ:

> **global metric + local metric + failure characterization**, không chỉ optimize rồi report chính energy mình optimize.

Phần này hiện draft làm đúng hướng khi dùng lDDT/RMSD độc lập thay vì NLL của chính Geometry.

---

# 18. Nếu làm đủ, RQ3 nên tách thành 3 câu

Hiện:

> “Geometry v2 có cải thiện local accuracy đồng thời preserve TM không?”



T thấy hơi nông.

T sẽ biến nó thành:

**RQ3a — Effectiveness**

> Does Geometry v2 improve local (C1') accuracy while preserving global fold quality?

**RQ3b — Mechanism**

> Which geometric priors and safeguards are responsible for the improvement?

**RQ3c — Generalization**

> Is the improvement consistent across unseen RNA families, prediction sources and initial-quality regimes?

Ba câu này chỉ xoay quanh **một method**, nhưng sâu hơn RQ3+RQ4 hiện tại nhiều.

---

# 19. Thesis structure mới t sẽ dựng thế này

| Chapter | Story                                                                |
| ------- | -------------------------------------------------------------------- |
| Ch1     | Problem, hypotheses, contributions                                   |
| Ch2     | RNA 3D, TBM, DL, refinement, metrics                                 |
| Ch3     | Data, temporal leakage, train/cal/test protocol                      |
| Ch4     | Time-safe TBM + hybrid candidate generation                          |
| Ch5     | **Geometry v2 method**                                               |
| Ch6     | TBM + hybrid component experiments                                   |
| Ch7     | **Geometry v2 ablations, validation, sensitivity, failure analysis** |
| Ch8     | Discussion + conclusion                                              |

GeoFuse:

**Appendix / Exploratory experiments**, khoảng 2–4 trang.

Hoặc Discussion có đúng một đoạn:

> preliminary experiments on local source-aware fusion did not produce reliable structural gains; learning coordinate corrections in local equivariant frames is therefore left for future work.

Xong.

**Không còn RQ4. Không còn Conv1D contribution. Không còn 100-RNA dataset bị bắt phục vụ một model thất bại.**

Nó được tái sử dụng để cứu Geometry validation.

---

# 20. Nếu thời gian hữu hạn, đây là thứ tự t sẽ bắt ní chạy

1. **Freeze GeoFuse.**
2. Chuyển 60/20/20 thành Geometry train/cal/test.
3. Chạy **Geometry v2 full trên 20 held-out RNA**.
4. Chạy Geometry ablation: source/backbone/angle/torsion/kink/adaptive.
5. Chạy simple-refinement baseline.
6. Chạy 2×2 factorial: TBM/hybrid × Geometry off/on.
7. Ablate TBM retrieval: MMseqs2/composite/union.
8. Ablate composite-score components hoặc ít nhất sensitivity.
9. Stratify Geometry theo source + initial quality.
10. Rewrite thesis **sau cùng**.

Nếu chỉ kịp **3 experiment mới**, t chọn:

[
\boxed{\text{Independent Geometry test}}
]

[
\boxed{\text{Geometry ablation}}
]

[
\boxed{\text{TBM/Hybrid}\times\text{Geometry 2×2}}
]

Ba cái này sẽ thay đổi mức độ thuyết phục của thesis nhiều nhất.

---

## Một điểm cuối t khá muốn ní để ý

Nếu bỏ GeoFuse thì **đừng còn nói “quality estimation is the next bottleneck” như central finding nữa**.

Abstract hiện vẫn nói selected 0.4713 vs oracle 0.5123 rồi:

> “qua đó xác định ước lượng chất lượng là nút thắt tiếp theo.”

Cái này lúc đó trở thành một **observation**, không phải thesis storyline.

Câu chuyện mạnh hơn sẽ là:

[
\boxed{
\text{better retrieval}
\rightarrow
\text{complementary candidate generation}
\rightarrow
\text{controlled local refinement}
}
]

và contribution chính cuối cùng là:

> **Geometry v2 improves local structural accuracy consistently, identifies the components responsible for the gain, preserves global fold quality, and generalizes to temporally/family-separated unseen RNA targets.**

Nếu ní làm được **đúng câu đó bằng experiments**, thì t nghĩ thesis từ bản hiện tại khoảng 7/10 có thể nhảy lên **8.5+ ở góc độ scientific design**, mà không cần phát minh thêm một architecture hoành tráng nào cả. Nó sẽ có một thứ mà rất nhiều thesis thiếu: **mỗi design choice đều có evidence cho biết nó đóng góp cái gì.**

[1]: https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003659 "DRfold2 is a deep learning-based tool that enables efficient and accurate RNA structure prediction | PLOS Biology"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6429776/?utm_source=chatgpt.com "QRNAS: software tool for refinement of nucleic acid structures"
[3]: https://academic.oup.com/nar/article/48/2/576/5651330 "oup.silverchair-cdn.com"
