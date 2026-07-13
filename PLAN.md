Chuẩn rồi ní. Top leaderboard confirm gần như 100% cái hướng mình đang converge về:

```text
Tìm template tốt
+ alignment/mapping tốt
+ dùng pretrained/DL khi template yếu hoặc thiếu
+ sửa/gap-fill/refine structure
+ chọn 5 predictions cuối
```

Và đúng như ní nói: **main contribution của mình sẽ nằm ở geometry-informed refinement/gap repair/source fusion**, không phải tự train một RNA 3D model từ đầu. Cái này sát thực tế, sát Kaggle, và không ảo.

Dưới đây là **final plan chốt để implement**, gồm cả repo WSL Linux và cách packaging thành notebook Kaggle chạy dưới 8h.

---

# 0. Chốt thesis direction cuối cùng

## Title đề xuất

**Temporal-Safe Hybrid Template-Based Modeling and Geometry-Informed Refinement for C1′-Level RNA 3D Structure Prediction**

Tiếng Việt:

**Dự đoán cấu trúc RNA 3D mức C1′ bằng mô hình lai template-based và tinh chỉnh hình học an toàn theo thời gian**

---

# 1. Thesis claim cuối cùng

Mình không claim:

```text
Tôi tạo model RNA folding mới từ đầu.
Tôi đánh bại mọi deep learning RNA model.
Tôi giải quyết full-atom RNA folding.
```

Mình claim:

```text
Tôi xây một pipeline lai cho Stanford RNA 3D Folding:

1. dùng temporal-safe template-based modeling để lấy fold information;
2. dùng pretrained/DL models như DRfold2/Boltz/Chai/RibonanzaNet-style prior khi template yếu;
3. dùng geometry-informed refinement để sửa gap, ổn định backbone,
   tránh clash, và dung hòa template với pretrained predictions;
4. sinh 5 cấu trúc C1′ cuối theo đúng format Kaggle.
```

Bài Kaggle yêu cầu output 5 bộ tọa độ C1′ cho mỗi RNA target, tức mỗi target cần 5 structures `[L, 3]`; proposal ban đầu của ní cũng đang đặt input là RNA sequence + MSA và output là 5 bộ tọa độ C1′.  

---

# 2. Final pipeline tổng thể

```text
Input:
- test_sequences.csv
- MSA/{target_id}.MSA.fasta
- PDB_RNA/
- pdb_seqres_NA.fasta
- pdb_release_dates_NA.csv
- pretrained model outputs/candidates nếu có

For each target RNA:

1. Read target sequence + temporal_cutoff
2. Search templates bằng MMseqs2
3. Filter templates theo release_date <= temporal_cutoff
4. Align target ↔ template
5. Transfer C1′ coordinates
6. Fill gaps/missing residues bằng geometry-aware reconstruction
7. Run pretrained/DL predictor nếu:
   - template yếu
   - template thiếu vùng lớn
   - còn thời gian inference
8. Refine candidate structures bằng geometry-informed optimizer
9. Generate 5 genuinely diverse structures
10. Write submission.csv
```

Kaggle final notebook phải chạy offline, không internet, xuất `submission.csv`, và giới hạn runtime CPU/GPU 8 giờ theo mô tả competition. 

---

# 3. Mình học gì từ top solutions?

## Từ 1st place

1st place theo text ní gửi là **Hybrid TBM + DRfold2**.

Điểm nên học:

```text
- Không train model từ đầu.
- Dùng TBM làm xương sống.
- Parse CIF thật kỹ, bao gồm modified nucleotides.
- Template search + alignment + coordinate transfer là trọng tâm.
- Gap filling không làm sơ sài.
- Refinement intensity phụ thuộc template confidence.
- DRfold2 dùng cho phần deep learning/fallback.
- Optimize ranking/selection/post-processing thay vì fine-tune model.
```

Cái cực quan trọng: họ cũng ưu tiên **overall fold** hơn atomic-level precision vì TM-score robust với local errors. Đây đúng với hướng của mình: **template/prior lấy global fold, geometry refinement sửa local issues**.

## Từ 2nd place

2nd place theo text ní gửi là **Deep Learning Representation Based TBM**.

Điểm nên học:

```text
- TBM vẫn là core.
- Dùng representation từ RibonanzaNet để search remote homologs.
- MMseqs2 clustering/template preparation là bắt buộc.
- Chai-1/Boltz-1 dùng khi template không đủ.
- Missing regions được patch bằng DL-predicted structures hoặc template khác.
- Không nhất thiết molecular dynamics/refinement nặng.
```

Cái này mở ra optional extension cho mình:

```text
RibonanzaNet representation-based search/alignment
```

Nhưng không nên cho vào core ngay từ ngày đầu, vì dễ nặng. Core vẫn là:

```text
MMseqs2 standard TBM
+ pretrained candidate fallback
+ geometry refinement
```

---

# 4. Contribution của mình khác gì top solutions?

Phải nói thật: **mình không nên claim chiến lược TBM + DL fallback là hoàn toàn mới**, vì top leaderboard đã chứng minh nó là hướng thắng.

Contribution nên đặt vào chỗ này:

```text
A confidence-weighted geometry-informed refinement framework
for repairing and reconciling template-derived and pretrained-predicted
C1′ RNA structures.
```

Cụ thể hơn:

```text
1. Template confidence controls refinement strength.
2. Gap residues get lower template weight and higher geometry/prior freedom.
3. Backbone distance constraints preserve C1′ chain continuity.
4. Clash and size constraints prevent physically implausible structures.
5. Pairwise distance/distogram prior, if available, is fused with template coordinates.
6. Best-of-5 generation includes real diversity:
   top templates, DL fallback, hybrid candidate, mirror hedge for distance-only branch.
7. Evaluation includes ablation:
   template-only vs template+refinement vs DL+refinement vs hybrid.
```

Tức là thesis không chỉ “copy top solution”, mà biến phần post-processing/refinement thành một **clean research module**, có ablation rõ, có temporal-safety rõ, có failure analysis rõ.

---

# 5. Method cuối cùng: 3 nguồn candidate

Với mỗi target, mình sinh candidate từ 3 nguồn.

## Source A — TBM candidates

```text
sequence
→ MMseqs2 search
→ exact alignment
→ temporal-safe template filtering
→ C1′ coordinate transfer
→ gap fill
→ template candidate
```

Đây là nguồn mạnh nhất nếu có template tốt.

## Source B — Pretrained/DL candidates

Có thể là:

```text
DRfold2
Boltz-1
Chai-1
RibonanzaNet/RibonanzaNet2-derived prior
```

Vai trò:

```text
- dùng khi template yếu;
- dùng để patch gap/missing regions;
- dùng làm alternative candidate trong best-of-5;
- dùng làm pairwise/contact/distance prior nếu model output hỗ trợ.
```

Trong 8 tuần, ưu tiên thực tế:

```text
Priority 1: chạy được DRfold2 hoặc một pretrained 3D predictor offline.
Priority 2: dùng output của nó làm candidate/gap patch.
Priority 3: nếu model có distance/distogram/contact prior thì dùng vào refinement loss.
```

Không nên đặt cược thesis vào việc train một contact model từ đầu.

## Source C — Geometry-informed refinement

Nhận input từ A/B rồi refine.

```text
candidate structure X0
+ template confidence
+ gap mask
+ optional pairwise prior
+ generic geometry priors
→ refined structure X*
```

---

# 6. Refinement objective cuối cùng

Không dùng 7 loss ngay. Dùng bản v1 gọn, chắc:

[
E(X)=
\lambda_{tpl}L_{tpl}
+\lambda_{dist}L_{dist}
+\lambda_{bb}L_{bb}
+\lambda_{clash}L_{clash}
+\lambda_{rg}L_{rg}
]

## 6.1. Template loss

Dùng khi candidate đến từ template:

[
L_{tpl} =
\sum_i w_i \lVert X_i - X^{tpl}_i \rVert^2
]

Trong đó:

```text
w_i cao:
- residue align tốt
- template coordinate reliable

w_i thấp:
- residue ở gap
- residue nội suy
- vùng missing
```

Đây là chỗ main contribution mạnh: **trust template where reliable, let geometry/prior repair where unreliable**.

---

## 6.2. Distance/distogram prior loss

Chỉ dùng nếu pretrained model có pairwise distance/distogram/contact usable.

Nếu có continuous distance:

[
L_{dist}
========

\sum_{|i-j|\ge k}
W_{ij}
SmoothL1(
\lVert X_i-X_j \rVert - D^{pred}_{ij}
)
]

Nếu có distogram:

[
L_{dist}
========

-\sum_{|i-j|\ge k}
W_{ij}
\log P_{ij}(bin(\lVert X_i-X_j \rVert))
]

Important:

```text
L_dist chỉ dùng cho non-local pairs, ví dụ |i-j| >= 3 hoặc 4.
Không để nó đánh nhau với backbone local distance.
```

---

## 6.3. Backbone C1′ distance loss

[
L_{bb}
======

\sum_i
\left(
\frac{
\lVert X_{i+1}-X_i \rVert-\mu_{bb}
}{
\sigma_{bb}
}
\right)^2
]

`μ_bb` và `σ_bb` lấy từ train labels.

Top solution cũng dùng khoảng C1′–C1′ consecutive quanh ~5.9 Å để gap fill; mình sẽ estimate từ data thay vì hard-code hoàn toàn.

---

## 6.4. Clash loss

[
L_{clash}
=========

\sum_{|i-j|>k}
ReLU(r_{min}-\lVert X_i-X_j \rVert)^2
]

Dùng để tránh RNA collapse thành một cục.

---

## 6.5. Radius of gyration loss

Dùng nhẹ để chống over-collapse hoặc over-expansion:

[
R_g(X)
======

\sqrt{
\frac{1}{L}
\sum_i
\lVert X_i-\bar{X} \rVert^2
}
]

[
L_{rg}
======

(R_g(X)-R_g^{target})^2
]

`R_g_target` có thể estimate theo length bins hoặc từ template/prior.

---

## 6.6. Angle/curvature để đâu?

Không đưa vào v1.

Để optional ablation:

```text
- angle loss
- unsigned curvature loss
- signed C1′ pseudo-dihedral/chirality term
```

Lý do: angle/curvature C1′ global distribution khá rộng, có thể làm hại nếu ép quá cứng.

---

# 7. Five predictions cuối cùng

## Nếu target có template tốt

```text
Prediction 1:
best template + conservative refinement

Prediction 2:
second structurally distinct template + refinement

Prediction 3:
best template + stronger gap repair / weaker template weight

Prediction 4:
DL/pretrained candidate + refinement

Prediction 5:
hybrid template + DL patch + refinement
```

## Nếu target template yếu hoặc không có template

```text
Prediction 1:
best pretrained/DL structure + refinement

Prediction 2:
mirror of distance-only/MDS-derived candidate nếu branch distance-only

Prediction 3:
alternative pretrained candidate / sampled prior

Prediction 4:
weak-template + low template weight + refinement

Prediction 5:
different optimization init / different prior confidence threshold
```

Important: phải đo diversity thật:

```text
pairwise self-TM giữa 5 predictions
```

Nếu 5 structures giống nhau >0.9 TM với nhau, best-of-5 gần như vô dụng.

---

# 8. Repo setup trong WSL Linux

Ní nên làm repo trong filesystem của WSL, không làm trực tiếp trên `/mnt/d` nếu project nhiều file nhỏ, vì I/O qua Windows mount chậm.

Ví dụ:

```bash
mkdir -p ~/projects
cd ~/projects
git clone <your_repo_url> rna3d-thesis
cd rna3d-thesis
```

Nếu data nằm ở Windows drive:

```bash
ln -s /mnt/d/Project/StanfordRNA/data ./data_raw
```

Nhưng code/cache nên để trong WSL:

```text
~/projects/rna3d-thesis/
```

---

# 9. Environment setup

## 9.1. System packages

```bash
sudo apt update
sudo apt install -y build-essential git wget curl unzip cmake pkg-config
```

## 9.2. Conda env

Ní đang dùng conda rồi, tạo env riêng:

```bash
conda create -n rna3d python=3.10 -y
conda activate rna3d
```

## 9.3. Core packages

```bash
conda install -y -c conda-forge -c bioconda \
  numpy pandas scipy scikit-learn biopython gemmi pyarrow tqdm \
  matplotlib seaborn jupyterlab ipykernel mmseqs2
```

## 9.4. PyTorch

Nếu WSL nhận GPU:

```bash
nvidia-smi
```

Sau đó cài PyTorch phù hợp CUDA. Nếu không chắc, bản đầu cứ CPU cũng được cho parser/TBM; refinement/DL mới cần GPU.

```bash
pip install torch torchvision torchaudio
```

Sau khi cài:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 9.5. Extra packages

```bash
pip install einops rich loguru
```

Có thể thêm sau:

```bash
pip install polars
```

nếu pandas/parquet chậm.

---

# 10. Repo structure cuối cùng

```text
rna3d-thesis/
│
├── README.md
├── environment.yml
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── paths.yaml
│   ├── refinement.yaml
│   ├── template_search.yaml
│   └── kaggle_inference.yaml
│
├── data/
│   ├── raw/                 # symlink hoặc local copy nhỏ
│   ├── interim/
│   ├── processed/
│   └── cache/
│
├── external/
│   ├── binaries/
│   │   ├── USalign
│   │   └── mmseqs
│   └── pretrained/
│       ├── drfold2/
│       ├── ribonanzanet/
│       ├── boltz/
│       └── chai/
│
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_scoring_us_align.ipynb
│   ├── 02_geometry_priors.ipynb
│   ├── 03_parse_pdb_rna.ipynb
│   ├── 04_template_search.ipynb
│   ├── 05_tbm_baseline.ipynb
│   ├── 06_pretrained_smoke_test.ipynb
│   ├── 07_refinement_demo.ipynb
│   ├── 08_ablation_analysis.ipynb
│   └── kaggle_submission.ipynb
│
├── src/
│   └── rna3d/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── io.py
│       │   ├── fasta.py
│       │   ├── labels.py
│       │   └── submission.py
│       │
│       ├── cif/
│       │   ├── parser.py
│       │   ├── nucleotide_map.py
│       │   └── extract_c1prime.py
│       │
│       ├── template/
│       │   ├── build_db.py
│       │   ├── mmseqs_search.py
│       │   ├── align.py
│       │   ├── transfer.py
│       │   ├── gap_fill.py
│       │   └── confidence.py
│       │
│       ├── pretrained/
│       │   ├── drfold2_runner.py
│       │   ├── boltz_runner.py
│       │   ├── chai_runner.py
│       │   ├── ribonanza_runner.py
│       │   └── prior_converter.py
│       │
│       ├── geometry/
│       │   ├── priors.py
│       │   ├── distances.py
│       │   ├── mds.py
│       │   ├── chirality.py
│       │   └── transforms.py
│       │
│       ├── refine/
│       │   ├── losses.py
│       │   ├── optimizer.py
│       │   ├── confidence_weights.py
│       │   └── generate_variants.py
│       │
│       ├── eval/
│       │   ├── usalign.py
│       │   ├── best_of_five.py
│       │   ├── self_tm.py
│       │   └── metrics.py
│       │
│       └── pipeline/
│           ├── run_target.py
│           ├── run_validation.py
│           └── run_inference.py
│
├── scripts/
│   ├── build_template_db.py
│   ├── build_mmseqs_db.sh
│   ├── run_tbm_baseline.py
│   ├── run_pretrained_smoke_test.py
│   ├── run_refinement.py
│   ├── run_ablation.py
│   └── make_kaggle_dataset.py
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── thesis_notes/
│
└── kaggle/
    ├── kaggle_submission.ipynb
    ├── inference_src/
    └── README_kaggle.md
```

---

# 11. Local development vs Kaggle inference

Đây là điểm rất quan trọng.

## Local WSL repo

Dùng để:

```text
- parse toàn bộ PDB_RNA
- build template DB
- build MMseqs2 DB
- setup pretrained models
- run validation
- run ablation
- viết thesis result
```

Local được phép mất thời gian hơn.

## Kaggle final notebook

Chỉ dùng để inference:

```text
- load precomputed template DB
- load MMseqs2 index
- load pretrained weights
- search templates nhanh
- generate/refine candidates
- write submission.csv
```

Không nên để Kaggle notebook:

```text
- parse toàn bộ CIF từ đầu
- train model
- build database từ scratch
- chạy ablation
- chạy US-align evaluation
```

Nói thẳng: **Kaggle notebook final phải là production inference**, không phải research notebook.

---

# 12. Precompute artifacts cần tạo trước khi lên Kaggle

Trong WSL/local:

```text
template_chains.parquet
template_coords.npz
template_metadata.parquet
mmseqs_db/
geometry_priors.json
nucleotide_mapping.json
pretrained_model_weights/
```

Rồi upload các thứ này thành Kaggle Dataset riêng, ví dụ:

```text
/kaggle/input/rna3d-precomputed/
    template_chains.parquet
    template_coords.npz
    geometry_priors.json
    mmseqs_db/
    binaries/
    pretrained/
```

Final notebook đọc:

```python
COMP_DATA = Path("/kaggle/input/stanford-rna-3d-folding")
PRECOMP = Path("/kaggle/input/rna3d-precomputed")
```

---

# 13. Implementation order cuối cùng

## Phase 1 — Scoring + sanity

Mục tiêu: chưa model gì cả, nhưng evaluation phải đúng.

Tasks:

```text
1. Load train/validation sequences and labels.
2. Handle missing coordinates, including sentinel values.
3. Create dummy 5-structure prediction.
4. Convert prediction to PDB-like/US-align-readable format.
5. Run US-align.
6. Implement best-of-5, multi-reference scoring.
7. Test:
   native vs native
   native vs rotated native
   native vs mirrored native
```

Deliverables:

```text
notebooks/01_scoring_us_align.ipynb
src/rna3d/eval/
dummy_score_table.csv
```

---

## Phase 2 — Geometry priors

Tasks:

```text
1. Extract C1′ coordinates from train labels.
2. Estimate adjacent C1′ distance mean/std.
3. Estimate clash threshold.
4. Estimate radius-of-gyration by length bins.
5. Save geometry_priors.json.
```

Deliverables:

```text
geometry_priors.json
adjacent_distance_distribution.png
rg_by_length.png
```

---

## Phase 3 — PDB_RNA parser + template DB

Tasks:

```text
1. Parse pdb_seqres_NA.fasta.
2. Parse pdb_release_dates_NA.csv.
3. Parse PDB_RNA/*.cif with gemmi.
4. Filter RNA chains, not DNA chains.
5. Map modified nucleotides to canonical bases.
6. Extract C1′ coordinates.
7. Save template DB.
```

Need be strict:

```text
- entity type = polyribonucleotide
- atom = C1′
- no blind A/C/G/U/T parsing
- handle modified residues
- handle missing/disordered C1′
```

Deliverables:

```text
template_chains.parquet
template_coords.npz
pdb_parse_report.csv
```

---

## Phase 4 — MMseqs2 template search

Tasks:

```text
1. Build MMseqs2 DB from template sequences.
2. For each target, search top candidate templates.
3. Exact align top candidates.
4. Filter by temporal_cutoff.
5. Apply self-leakage guard for validation.
6. Rank by identity, coverage, gap ratio.
```

Template confidence:

```text
conf = identity × coverage × completeness × temporal_validity
```

Deliverables:

```text
template_hits_validation.csv
template_quality_bins.csv
```

---

## Phase 5 — TBM coordinate transfer

Tasks:

```text
1. Align target-template.
2. Copy coordinates for matched residues.
3. Gap fill:
   - short internal gap: interpolation
   - longer gap: interpolation + sinusoidal/perpendicular perturbation
   - terminal missing: extend along backbone direction
4. Create mask_template and residue confidence.
5. Generate top-1 and top-5 TBM candidates.
```

Deliverables:

```text
tbm_top1_predictions.csv
tbm_top5_predictions.csv
tbm_baseline_score.csv
```

---

## Phase 6 — Pretrained/DL smoke test

Đưa lên sớm, không để tuần 6.

Tasks:

```text
1. Chọn pretrained candidates:
   - DRfold2 first
   - optionally Boltz-1/Chai-1
   - optionally RibonanzaNet representation/prior

2. Verify:
   - load offline được không
   - runtime thế nào
   - output là 3D coordinate hay pairwise prior
   - output có map về C1′ không
   - có chạy được trong Kaggle không
```

Decision gate:

```text
Nếu pretrained model chạy ổn:
    dùng làm fallback/candidate/gap patch.

Nếu không:
    thesis core vẫn là TBM + geometry refinement,
    no-template case ghi là limitation.
```

Deliverables:

```text
pretrained_smoke_test_report.md
pretrained_candidate_outputs/
```

---

## Phase 7 — Geometry-informed refinement

Implement v1 loss:

```text
L_tpl
L_dist hoặc distogram potential nếu có
L_bb
L_clash
L_rg
```

Tasks:

```text
1. Refine TBM candidates.
2. Refine pretrained candidates.
3. Refine hybrid template+pretrained candidates.
4. Compare before vs after.
5. Tune confidence-based strength.
```

Adaptive refinement:

```text
high template confidence:
    low refinement strength

medium confidence:
    moderate refinement

low confidence:
    stronger backbone/clash/prior terms
```

Deliverables:

```text
refined_predictions.csv
before_after_refinement_table.csv
```

---

## Phase 8 — Five-structure generator

Candidate pool:

```text
- top TBM candidates
- refined TBM candidates
- pretrained candidates
- hybrid patched candidates
- mirror hedge for distance-only candidate
```

Selection criteria:

```text
1. template confidence
2. pretrained confidence/energy if available
3. geometry energy
4. structural diversity by self-TM
```

Output exactly 5 structures per target.

Deliverables:

```text
five_structure_predictions.csv
self_tm_diversity_table.csv
```

---

## Phase 9 — Experiments and ablation

Main baselines:

```text
B0: dummy helix/straight chain
B1: TBM top-1
B2: TBM top-5
B3: pretrained/DL candidate only
B4: TBM + geometry refinement
B5: pretrained + geometry refinement
B6: hybrid TBM + pretrained + geometry refinement
```

Ablations:

```text
- no refinement
- no gap-aware weights
- no clash
- no radius-of-gyration
- no pretrained candidate
- no mirror hedge
- best-of-1 vs best-of-5
- no temporal filtering, only as leakage demonstration
```

Primary analysis:

```text
high template quality
medium template quality
low/no template
short vs long RNA
deep vs shallow MSA
```

Expected result:

```text
- high-template: TBM already strong, refinement small gain or neutral
- medium-template: refinement helps most
- no-template: performance depends mainly on pretrained/DL prior
```

---

# 14. Kaggle notebook structure

Final `kaggle_submission.ipynb` nên có các section:

```text
1. Imports and paths
2. Load test_sequences.csv
3. Load precomputed template DB
4. Load geometry priors
5. Load pretrained model if available
6. For each target:
   a. template search
   b. TBM candidates
   c. pretrained candidates if needed/time allows
   d. refinement
   e. select 5 structures
7. Validate submission format
8. Save /kaggle/working/submission.csv
```

Notebook không nên chứa training/ablation.

Pseudo:

```python
for target in test_targets:
    seq = target.sequence
    cutoff = target.temporal_cutoff

    templates = search_templates_mmseqs(seq, cutoff)
    tbm_candidates = build_tbm_candidates(seq, templates)

    dl_candidates = []
    if should_run_pretrained(templates, seq):
        dl_candidates = run_pretrained_candidates(seq)

    candidate_pool = tbm_candidates + dl_candidates

    refined_pool = []
    for cand in candidate_pool:
        refined = refine_candidate(cand, geometry_priors)
        refined_pool.append(refined)

    five = select_five(refined_pool, diversity=True)
    writer.add_target(target, five)

writer.save("/kaggle/working/submission.csv")
```

---

# 15. Runtime strategy để dưới 8h

Do competition notebook cần chạy trong giới hạn runtime, mình phải thiết kế inference kiểu tiết kiệm. 

## Không làm trong Kaggle inference

```text
- Không parse toàn bộ CIF.
- Không build MMseqs DB.
- Không train model.
- Không chạy US-align.
- Không chạy ablation.
```

## Làm trong Kaggle inference

```text
- MMseqs2 search trên DB đã build sẵn.
- Exact alignment top candidates.
- Load C1′ coords từ cache.
- Run pretrained only when needed.
- Refine limited steps.
- Write submission.
```

## Runtime knobs

```text
MAX_TEMPLATES = 5 hoặc 10
MAX_REFINEMENT_STEPS = 100–300
RUN_PRETRAINED_ONLY_IF_TEMPLATE_CONF < threshold
SKIP_EXPENSIVE_MODEL_FOR_SHORT_GOOD_TEMPLATE_TARGETS = True
```

---

# 16. Thesis methodology final wording

Câu này nên dùng trong thesis:

> Inspired by the top leaderboard strategies, the proposed method follows a hybrid paradigm in which template-based modeling provides the primary fold information when homologous structures are available, while pretrained RNA structure predictors provide fallback candidates for weak-template targets. The thesis contribution focuses on a confidence-weighted geometry-informed refinement module that repairs missing template regions, preserves C1′ backbone consistency, prevents steric clashes, and reconciles template-derived coordinates with pretrained structural priors. The final system produces five diverse C1′ coordinate predictions per RNA target under the Stanford RNA 3D Folding benchmark.

Tiếng Việt dễ hiểu:

> Phương pháp của luận văn là một pipeline lai: dùng template khi tìm được cấu trúc tương tự, dùng pretrained model khi template yếu hoặc thiếu, sau đó áp dụng một module tinh chỉnh hình học có trọng số theo độ tin cậy để sửa vùng thiếu, giữ tính liên tục của backbone C1′, tránh va chạm không hợp lý và kết hợp thông tin từ template với mô hình pretrained. Hệ thống cuối cùng sinh ra 5 cấu trúc C1′ cho mỗi RNA target theo đúng benchmark Stanford RNA 3D Folding.

---

# 17. Minimum viable thesis và strong thesis

## Minimum viable thesis

Nếu pretrained model khó setup:

```text
- temporal-safe TBM
- robust CIF parser
- MMseqs2 template search
- coordinate transfer
- gap filling
- geometry refinement
- ablation on validation
```

Claim:

```text
Geometry-informed refinement improves or stabilizes template-derived RNA structures,
especially in medium-confidence template cases.
```

## Strong thesis

Nếu pretrained model chạy ổn:

```text
- TBM + DRfold2/Boltz/Chai candidates
- geometry refinement for both
- hybrid patching of missing template regions
- best-of-5 selection with diversity
```

Claim:

```text
Hybrid TBM + pretrained candidates + geometry-informed refinement
outperforms TBM-only and pretrained-only baselines.
```

---

# 18. Từ giờ mình chốt phương án nào?

Chốt như này:

```text
Core mandatory:
1. MMseqs2-based temporal-safe TBM
2. robust C1′ extraction from CIF
3. template-to-target alignment and coordinate transfer
4. geometry-aware gap filling
5. confidence-weighted geometry refinement
6. best-of-5 output generation
7. Kaggle-compatible notebook

Strong extension:
8. DRfold2/pretrained candidate fallback
9. hybrid template + pretrained patching
10. optional RibonanzaNet representation-based search
```

Không còn đổi hướng sang train model mới.

Không còn claim geometry tự fold RNA.

Không còn để MMseqs2/pretrained smoke test quá muộn.

---

# 19. Lịch làm thực tế 8 tuần sau khi chốt

```text
Week 1:
Scoring, data audit, US-align, dummy submission.

Week 2:
CIF parser, geometry priors, pretrained smoke test, MMseqs2 install.

Week 3:
Template DB cache, MMseqs2 search, temporal-safe template ranking.

Week 4:
Coordinate transfer, gap filling, TBM top-1/top-5 baseline.

Week 5:
Geometry refinement v1, before/after evaluation.

Week 6:
Pretrained candidates + hybrid patching + five-structure generation.

Week 7:
Ablations, diversity self-TM, template-quality analysis.

Week 8:
Kaggle notebook packaging, runtime optimization, thesis writing.
```

---

# 20. Chốt hạ

Đúng, mình sẽ chốt phương án:

```text
Hybrid TBM + pretrained/DL fallback + geometry-informed refinement
```

Trong đó:

```text
TBM/template:
    nguồn fold chính khi có homolog.

Pretrained/DL:
    nguồn fold/candidate khi template yếu hoặc thiếu.

Geometry refinement:
    main contribution của thesis:
    sửa gap, repair local geometry, tránh clash,
    fusion template với pretrained structures,
    tạo candidates sạch hơn cho best-of-5.
```

Kaggle final alignment:

```text
Local WSL:
    build, parse, cache, validate, ablation.

Kaggle notebook:
    load cache + models,
    run inference only,
    generate 5 C1′ predictions,
    save submission.csv under 8h.
```

Từ giờ nên bắt đầu bằng **Week 1: scoring + data audit + repo setup**, rồi khóa dần từng module. Không thêm ý tưởng mới vào core nữa; chỉ thêm nếu experiment chứng minh cần.

___________________


Tớ nghĩ cảm giác của ní **đúng một nửa**: project không yếu, nhưng framing hiện tại “top solutions làm TBM + pretrained, còn mình thêm geometry refinement” vẫn chưa đủ sắc.

Vấn đề nằm ở chỗ kết quả của chính mình đã cho thấy:

* tăng TM từ `0.212 → 0.307` chủ yếu đến từ **cải thiện template search recall**;
* gradient refinement gần như **TM-neutral**;
* nó giảm clash và backbone deviation, nhưng lại làm sharp kinks tăng gần gấp đôi.

Vậy không nên tiếp tục claim rằng “geometry refinement nói chung” là contribution. Contribution cần được nâng lên thành:

> **Một phương pháp confidence-aware để kết hợp template và pretrained predictions theo từng vùng, sau đó chiếu cấu trúc lên một manifold hình học RNA phụ thuộc motif.**

Đây mới là một methodology có câu chuyện: top solutions đã chứng minh A+B hiệu quả, nhưng integration/refinement của họ còn heuristic; mình giải quyết điểm đó.

---

# 1. Research gap nên viết như thế nào?

## Các phương pháp top đã chứng minh

Từ writeup và reproduction của mình:

### Method A — Template-based modeling

```text
search template
→ alignment
→ coordinate transfer
→ gap filling
```

Mạnh khi tìm được template phù hợp.

### Method B — Pretrained/deep-learning predictors

```text
sequence / MSA
→ DRfold2 / Boltz / Chai / RNA foundation model
→ 3D candidate hoặc pairwise prior
```

Dùng cho:

* weak/no-template cases;
* vùng template bị missing;
* tạo candidate diversity.

Các pretrained RNA models có thể cung cấp không chỉ tọa độ mà còn secondary-structure hoặc inter-helical information; RhoFold+ là một ví dụ end-to-end dự đoán RNA 3D đồng thời có các output cấu trúc trung gian như secondary structure và inter-helical angles. ([arXiv][1])

### Method C — Heuristic post-processing

Top-1 dùng rule-based nudging:

* sửa sequential distance;
* đẩy clash;
* thêm base-pair attraction nhẹ;
* scale bằng global template confidence.

Top-2 chủ yếu patch vùng missing bằng template hoặc DL candidate sau structural alignment.

---

## Limitation thật sự

Phần integration/refinement hiện vẫn có bốn điểm yếu.

### 1. Quyết định ở mức whole structure

Thông thường pipeline chọn:

```text
dùng template
hoặc
dùng DL
```

hoặc patch DL vào đúng vùng template thiếu.

Nhưng trong một structure có thể tồn tại đồng thời:

```text
stem A: template rất đáng tin
loop B: template sai alignment
junction C: DL tốt hơn
terminal D: cả hai đều không chắc
```

Không nên bắt toàn bộ target tin vào cùng một nguồn.

### 2. Confidence quá thô

Global score kiểu:

```text
confidence = identity × coverage × completeness
```

không nói được residue nào đáng tin.

Hai template có cùng confidence tổng thể nhưng có thể:

* một cái sai rải đều;
* một cái đúng 90% và sai nghiêm trọng ở một junction.

Refinement strength cần là `per-residue`, không chỉ `per-target`.

### 3. Geometry rules không phụ thuộc structural context

Một mean angle hoặc backbone rule dùng cho mọi residue là quá thô.

RNA có:

* stem;
* hairpin loop;
* bulge;
* internal loop;
* junction;
* unpaired terminal.

Các vùng này có phân phối góc và torsion rất khác nhau. Kết quả kink hiện tại chính là bằng chứng: optimizer thỏa distance bằng cách bẻ góc quá mạnh.

Các nghiên cứu hình học RNA gần đây cũng nhấn mạnh kiến trúc RNA mang tính motif và representation hình học có thể mã hóa motif-level organization, thay vì coi toàn bộ backbone như một loại đường cong đồng nhất. ([arXiv][2])

### 4. TM-score không đánh giá hết chất lượng refinement

TM-score ưu tiên global fold. Một model có thể có cùng TM nhưng:

* nhiều clash hơn;
* kink nhiều hơn;
* geometry backbone tệ hơn;
* gap reconstruction thiếu ổn định.

Các framework RNA 3D gần đây dùng cả metric global và RNA-specific local validity/self-consistency thay vì chỉ một structure similarity score. ([arXiv][3])

---

# 2. Methodology đề xuất: **GeoFuse-RNA**

## Tên đầy đủ

**GeoFuse-RNA: Confidence-Aware Segment-Level Fusion and Motif-Conditioned Geometric Refinement for Hybrid RNA 3D Prediction**

Tiếng Việt:

**GeoFuse-RNA: Kết hợp cấu trúc theo độ tin cậy từng vùng và tinh chỉnh hình học phụ thuộc motif cho dự đoán RNA 3D lai**

Đây nên là main method của thesis.

---

# 3. Ý tưởng trung tâm

Thay vì:

```text
TBM candidate
→ global geometry refine
```

mình làm:

```text
TBM candidates + pretrained candidates
        ↓
phân cụm theo global fold
        ↓
ước lượng độ tin cậy của từng nguồn tại từng residue
        ↓
fuse các nguồn theo từng segment
        ↓
motif-conditioned differentiable geometry projection
        ↓
quality–diversity selection
        ↓
5 structures
```

Tức là pretrained model và TBM vẫn cung cấp **fold information**, nhưng contribution của mình là:

> biết tin nguồn nào, ở vùng nào, với mức độ bao nhiêu, và làm sao dung hòa chúng mà không tạo seam, clash hoặc kink.

---

# 4. Step-by-step methodology

## Step 1 — Candidate generation

Tạo candidate bank từ nhiều nguồn.

### Template candidates

```text
MMseqs2 search
+ composite-similarity search
→ exact alignment
→ coordinate transfer
→ gap filling
```

Giữ top-K, ví dụ 5–10 candidate.

### Pretrained candidates

Ưu tiên thực tế:

```text
DRfold2:
    3D structure candidates

RibonanzaNet2:
    representation / secondary / pairwise prior
    tùy checkpoint thực tế cung cấp gì

Boltz/Chai:
    optional candidate cho target ngắn hoặc chạy Kaggle GPU
```

Không cần train model 3D từ đầu.

### De novo candidate

Giữ de novo branch làm diversity/fallback, nhưng không coi nó là nguồn chính khi pretrained model có sẵn.

---

## Step 2 — Fold-family clustering

Không được fuse hai structure có global folds hoàn toàn khác nhau.

Tính self-TM hoặc pairwise structural similarity giữa candidates:

```text
candidate 1 ─┐
candidate 2 ─┼─ fold family A
candidate 3 ─┘

candidate 4 ─┐
candidate 5 ─┴─ fold family B
```

Mỗi cluster đại diện cho một hypothesis về global fold.

Sau đó chỉ fuse candidates **trong cùng cluster**.

Điều này tránh trường hợp:

```text
average hai fold khác topology
→ structure trung gian vô nghĩa
```

Best-of-5 cuối cùng có thể lấy từ nhiều fold clusters khác nhau.

---

## Step 3 — Per-residue source confidence

Với mỗi source `s`, residue `i`, tính:

[
q_{i,s}\in[0,1]
]

### Template confidence features

* global sequence identity;
* alignment coverage;
* local alignment score;
* match/mismatch;
* gap mask;
* coordinate resolved hay interpolated;
* template completeness;
* agreement với các template khác;
* khoảng cách tới alignment boundary.

Ví dụ:

```text
exact match, resolved C1′, templates agree
→ q cao

gap-filled, mismatch, templates disagree
→ q thấp
```

### Pretrained confidence features

Tùy model:

* pLDDT-like confidence;
* PAE/distogram entropy;
* variance giữa nhiều model samples;
* local agreement với other candidates;
* base-pair/secondary-structure confidence.

### Hai phiên bản

#### Version cơ bản

Dùng công thức rule-based để tạo `q`.

#### Version mạnh — contribution ML nhẹ

Train một **confidence gate** nhỏ:

```text
alignment features
+ sequence features
+ local geometry
+ source disagreement
→ predicted per-residue reliability q_i,s
```

Model chỉ cần:

* 1D CNN;
* BiLSTM;
* hoặc tiny Transformer.

Không cần SE(3)-equivariant model lớn vì nó không trực tiếp predict coordinates, chỉ predict mức độ tin cậy.

---

# 5. Train confidence gate như thế nào?

Dùng `train_v2` temporal-safe.

## Tạo pseudo-template training examples

Từ native structures:

* thêm coordinate noise;
* xóa residue blocks;
* tạo alignment gaps;
* shift một segment;
* thay một đoạn bằng template khác;
* tạo terminal extension;
* inject local clash;
* inject sharp kink.

Ngoài synthetic corruptions, dùng real template-target pairs nếu tìm được homolog trước cutoff.

## Label

Sau khi align candidate với native bằng Kabsch:

[
e_{i,s}=|X^{source}_{i,s}-X^{native}_i|
]

Có thể định nghĩa:

[
q^{target}_{i,s}
================

\exp\left(-\frac{e_{i,s}^2}{2\sigma^2}\right)
]

Hoặc classification:

```text
good residue: local error < 3 Å
bad residue:  local error ≥ 3 Å
```

Gate học được:

> với alignment/context như thế này, residue của template hoặc pretrained source đáng tin đến đâu.

Đây là phần làm methodology khác rõ so với top solution rule-based.

---

# 6. Segment-level fusion

Sau khi candidates trong cùng fold cluster đã được structural alignment về cluster medoid, tạo fused scaffold.

Naive weighted average:

[
X_i^{fused}
===========

\frac{\sum_s q_{i,s}X_{i,s}}
{\sum_s q_{i,s}}
]

Nhưng không nên chỉ average rồi kết thúc.

Fused coordinates là initialization; optimizer sẽ giải quyết:

* seam giữa các source;
* backbone discontinuity;
* clashes;
* incompatible local distances.

Có thể smooth confidence theo segment để tránh source switching từng residue:

```text
residues 1–25: template A
residues 26–45: pretrained model
residues 46–80: template A
```

thay vì đổi nguồn liên tục từng nucleotide.

---

# 7. Motif-conditioned geometry refinement

Đây là geometry contribution được nâng cấp.

## Structural context

Dự đoán hoặc suy ra mỗi residue thuộc:

```text
stem
hairpin loop
internal loop / bulge
junction
unpaired / terminal
```

Bản đầu có thể chỉ dùng:

```text
paired
unpaired
```

Từ train structures, học phân phối geometry theo context:

[
p_m(d,\theta,\tau)
]

trong đó:

* (d): adjacent C1′ distance;
* (\theta): pseudo-bond angle của ba C1′ liên tiếp;
* (\tau): signed pseudo-dihedral của bốn C1′ liên tiếp;
* (m): motif/context class.

Thay vì ép:

```text
mọi residue → cùng một mean angle
```

mình dùng:

```text
stem residue → stem geometry distribution
loop residue → loop geometry distribution
junction → broad flexible distribution
```

---

## Energy function

Với một fold cluster:

[
X^*
===

\arg\min_X E(X)
]

[
E(X)
====

\lambda_{src}L_{source}
+
\lambda_{pair}L_{pair}
+
\lambda_{geom}L_{motif}
+
\lambda_{clash}L_{clash}
+
\lambda_{size}L_{Rg}
]

### Source-consistency loss

[
L_{source}
==========

\sum_i\sum_s
q_{i,s},
\rho\left(
|X_i-X_{i,s}|
\right)
]

Dùng robust loss như Huber để một source sai không kéo structure quá mạnh.

### Pairwise pretrained prior

Nếu model trả distogram:

[
L_{pair}
========

-\sum_{i,j}w_{ij}
\log
P_{ij}\left(
bin(|X_i-X_j|)
\right)
]

Nếu chỉ có coordinates, dùng distance consensus từ pretrained candidates.

### Motif-conditioned geometry

[
L_{motif}
=========

-\sum_i
\log p_{m_i}(d_i,\theta_i,\tau_i)
]

Điểm này sửa đúng bug v1:

```text
v1 giảm distance error bằng cách tạo kink
v2 phạt geometry joint bất thường
```

### Clash loss

Giữ non-local C1′ pairs khỏi overlap.

### Rg/size loss

Dùng nhẹ để chống collapse hoặc overexpansion.

---

# 8. Stage-wise optimization

Không optimize mọi thứ cùng lúc với cùng weight.

## Stage A — Preserve/fuse global fold

```text
source loss mạnh
pairwise pretrained prior mạnh
geometry nhẹ
```

Mục tiêu:

* giữ topology từ TBM/DL;
* dung hòa các nguồn.

## Stage B — Repair uncertain regions

```text
freeze hoặc gần-freeze high-confidence residues
optimize gap/low-confidence residues mạnh hơn
```

## Stage C — Geometry projection

```text
motif angle/torsion
backbone
clash
Rg
```

Mục tiêu:

* xóa seam;
* giảm clash;
* không tạo kink.

Đây là khác biệt rõ với single-pass rule nudging.

---

# 9. Sinh 5 predictions

Mỗi fold cluster tạo ít nhất một refined candidate.

Ví dụ:

```text
S1: strongest TBM cluster, conservative fusion
S2: strongest TBM cluster, pretrained-heavy fusion
S3: second fold cluster
S4: pretrained-only cluster
S5: uncertainty sample / third fold family
```

Nếu distance-only branch có reflection ambiguity thì dùng mirror hedge.

Diversity không đến từ đổi nhẹ lambda, mà đến từ:

* distinct fold clusters;
* distinct source mixtures;
* sampled distograms;
* distinct confidence hypotheses.

---

# 10. Candidate selection

Ở test time không có TM-score, nên selector dùng:

```text
source confidence
pretrained confidence
geometry likelihood
clash/kink penalties
cluster support
structural diversity
```

Một score mẫu:

[
Q(X)=
aQ_{source}
+bQ_{pretrained}
-cE_{geom}
-dE_{clash}
-eU_{source}
]

Nhưng không chọn top-5 score thuần vì có thể toàn cùng fold.

Dùng quality-diversity selection:

```text
chọn candidate chất lượng cao
nhưng phạt candidate quá giống các candidate đã chọn
```

---

# 11. Cái này unique ở đâu?

Không claim:

> TBM + pretrained là mới.

Không claim:

> geometry constraints là mới.

Claim:

### Contribution 1 — Per-residue source fusion

Top pipelines chủ yếu chọn whole candidate hoặc patch gaps. Mình fuse template và pretrained predictions theo độ tin cậy từng residue/segment.

### Contribution 2 — Learned confidence gating

Mình học “nơi nào nên tin nguồn nào” từ temporal-safe corruption-repair data, thay vì dùng một global threshold.

### Contribution 3 — Motif-conditioned geometry projection

Không dùng global C1′ mean rules. Geometry phụ thuộc paired/unpaired hoặc stem/loop/junction context.

### Contribution 4 — Fold-cluster-aware best-of-5

Không hòa trộn incompatible folds; mỗi fold hypothesis được refine riêng và giữ diversity.

### Contribution 5 — Honest multi-objective evaluation

Không chỉ report TM. Report cả:

* clashes;
* backbone deviation;
* sharp kinks;
* torsion likelihood;
* gap repair;
* confident-region drift;
* runtime;
* temporal leakage.

Ở mức Master thesis, đây là contribution đủ rõ và khó bị nói là “chỉ recreate leaderboard”.

---

# 12. Research questions

## RQ1

**Can per-residue confidence-aware fusion of template and pretrained predictions improve RNA 3D accuracy compared with selecting either source as a whole?**

## RQ2

**Can motif-conditioned geometric projection reduce clashes, backbone distortion, and sharp kinks simultaneously while preserving the global fold?**

## RQ3

**Does fold-family-aware quality–diversity selection improve best-of-five TM-score compared with ranking candidates independently?**

## RQ4

**Which template-confidence regimes benefit most from fusion and refinement?**

---

# 13. Hypotheses

### H1 — Accuracy

GeoFuse-RNA sẽ tăng best-of-5 TM so với:

```text
TBM only
pretrained only
raw union TBM + pretrained
rule-based refinement
gradient v1
```

Gain dự kiến chủ yếu đến từ:

* medium-confidence templates;
* missing segments;
* disagreements giữa template và pretrained source.

### H2 — Geometry

GeoFuse-RNA giữ được:

```text
clash ↓
backbone deviation ↓
```

nhưng không làm:

```text
sharp kinks ↑
torsion likelihood xấu đi
```

### H3 — Reliability

High-confidence template residues sẽ di chuyển ít hơn gap/low-confidence residues.

### H4 — Selection

Fold-cluster selection sẽ tăng best-of-5 gain và giữ self-TM diversity tốt hơn lấy năm candidates có score cao nhất.

---

# 14. Experiments bắt buộc

## Main comparison

| Method                       | Ý nghĩa                                 |
| ---------------------------- | --------------------------------------- |
| Top-1 reproduced             | strong leaderboard-derived TBM baseline |
| Our TBM + composite          | current strong baseline                 |
| Pretrained only              | DL source                               |
| Raw TBM + pretrained union   | candidate generation benefit            |
| Rule-based refinement        | top-1 refinement baseline               |
| Gradient v1                  | current method                          |
| GeoFuse heuristic confidence | proposed without learned gate           |
| GeoFuse learned confidence   | full method                             |

## Ablation

| Ablation                  | Câu hỏi                                |
| ------------------------- | -------------------------------------- |
| no per-residue confidence | confidence gating có ích không         |
| no pretrained source      | fusion có hơn TBM không                |
| no template source        | phụ thuộc TBM đến đâu                  |
| no motif conditioning     | global geometry vs contextual geometry |
| no angle/torsion          | kink có quay lại không                 |
| no clustering             | fuse incompatible folds có hại không   |
| whole-candidate selection | segment fusion có ích không            |
| single-stage optimization | stage-wise có ổn định hơn không        |
| best-of-1                 | best-of-5 thực sự giúp bao nhiêu       |

---

# 15. Metrics để defense chắc

## Accuracy

* mean best-of-5 TM;
* mean top-1 TM;
* per-target paired ΔTM;
* bootstrap confidence interval;
* number of targets improved.

## Candidate-generation ceiling

### Oracle pool TM

```text
best TM trong toàn bộ candidate pool
```

Cho biết nguồn candidate có chứa đúng fold hay chưa.

### Selection regret

[
regret =
TM_{oracle\ pool}
-----------------

TM_{selected\ best5}
]

Phân biệt:

```text
candidate generation yếu
vs
selection yếu
```

## Geometry

* clashes/residue;
* adjacent backbone deviation;
* sharp-kink rate;
* motif-conditioned angle NLL;
* signed pseudo-torsion NLL;
* relative Rg error.

## Fusion behavior

* confident-residue drift;
* gap-residue movement;
* gap-region RMSD;
* source usage theo segment;
* geometry seam error tại source boundaries.

## Diversity

* mean/min/max self-TM;
* số fold clusters được giữ trong final five;
* best-of-5 gain over best-of-1.

## Practicality

* runtime/target;
* peak VRAM;
* full notebook runtime;
* failure/fallback rate.

---

# 16. Success criteria cụ thể

Thay vì viết “expect score cao hơn”, đặt tiêu chí kiểm chứng:

1. Full method có best-of-5 TM cao hơn `0.307` current temporal-safe pipeline.
2. Full method cao hơn raw TBM+pretrained candidate union, chứng minh fusion/refinement có giá trị riêng.
3. Gradient v2 giữ hoặc giảm clash/backbone so với v1 nhưng sharp-kink không cao hơn no-refinement.
4. Medium-confidence/gapped targets có paired improvement.
5. Selection regret giảm.
6. Full Kaggle inference dưới 8 giờ.

Nếu pretrained candidates không nâng oracle pool TM, không refiner nào cứu được. Khi đó conclusion trung thực là bottleneck nằm ở candidate generation.

---

# 17. Implementation roadmap thực tế

## Phase A — Pretrained branch

* chạy DRfold2;
* kiểm tra RibonanzaNet2 checkpoint/output;
* cache 3D candidates và confidence/prior;
* đo oracle pool TM.

Đây là gate đầu tiên.

## Phase B — Geometry v2

* thêm angle distribution;
* thêm signed pseudo-torsion;
* paired/unpaired conditioning trước;
* run v1 vs v2 ablation.

## Phase C — Fold clustering + heuristic fusion

* align candidates;
* cluster by self-TM;
* rule-based per-residue confidence;
* source fusion;
* stage-wise optimization.

Đây đã đủ thành full non-learned method.

## Phase D — Learned confidence gate

* generate corruption dataset;
* train tiny 1D model;
* replace heuristic confidence;
* compare learned vs heuristic.

## Phase E — Selection + submission

* quality-diversity selector;
* runtime profiling;
* Kaggle late submission;
* thesis tables.

---

# 18. Thesis positioning cuối cùng

Title nên đổi thành:

> **GeoFuse-RNA: Confidence-Aware Fusion of Template and Pretrained Predictions with Motif-Conditioned Geometric Refinement for RNA 3D Structure Prediction**

Main contribution statement:

> Existing competitive approaches demonstrate that template-based modeling and pretrained RNA predictors provide strong and complementary fold hypotheses. However, their combination is commonly performed through whole-model fallback, missing-region patching, or fixed rule-based refinement. This thesis proposes GeoFuse-RNA, a model-agnostic framework that clusters candidate structures by fold, estimates source reliability at residue level, fuses template and pretrained predictions segment-wise, and projects the resulting structures onto context-dependent RNA geometric distributions through differentiable optimization.

# 19. Chốt thật lòng

**Geometry refinement v1 hiện tại chưa đủ làm “đinh” của thesis.** Kết quả kink đã chứng minh điều đó.

Nhưng đấy không phải ngõ cụt; nó chính là research gap:

```text
fixed distance-based refinement
→ sửa một metric
→ làm hỏng metric khác
```

Method mới giải quyết bằng:

```text
per-residue confidence
+ source fusion
+ motif-conditioned distance/angle/torsion geometry
+ fold-aware diversity
```

Và điểm TM cao hơn, nếu có, sẽ đến từ **fusion TBM + pretrained predictions**, còn geometry projection làm cho fusion ổn định, hợp lý và không tạo artifact.

Đó là một story đủ mạnh:

> Top solutions prove candidate sources A and B are effective. Their public integration remains largely heuristic. We propose an adaptive, confidence-aware, geometrically principled fusion layer, and evaluate both fold accuracy and structural validity under temporal-safe conditions.

[1]: https://arxiv.org/abs/2207.01586?utm_source=chatgpt.com "Accurate RNA 3D structure prediction using a language model-based deep learning approach"
[2]: https://arxiv.org/abs/2603.19636?utm_source=chatgpt.com "RiboSphere: Learning Unified and Efficient Representations of RNA Structures"
[3]: https://arxiv.org/abs/2406.13839?utm_source=chatgpt.com "RNA-FrameFlow: Flow Matching for de novo 3D RNA Backbone Design"
