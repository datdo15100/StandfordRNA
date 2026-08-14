Ừa, t bắt được cái **“chất”** ní muốn rồi. Và t nghĩ đây là một hướng sửa rất hợp cho thesis hiện tại.

Paper mẫu có một kiểu văn khá rõ: **formal, compact, confident nhưng không phô trương; mỗi đoạn đều có mục đích; methodology được kể theo logic “problem → design choice → mechanism”; results được kể theo “observation → comparison → interpretation”**. Abstract của nó chẳng hạn đi rất thẳng: nêu problem → giới thiệu framework → mô tả hai component → đưa số → nêu nuance giữa heuristic/LSTM → kết thúc bằng practical implication.  Introduction cũng chia paragraph rất sạch: problem, gap, prior work, proposed solution, contributions.

Từ giờ, nếu sửa thesis của ní theo style này, t sẽ coi đây là **house style**:

### 1. Mỗi paragraph chỉ làm một việc

Paper mẫu rất ít kiểu paragraph chứa 5–6 ý ngang hàng. Thường nó có cấu trúc:

**topic sentence → explanation → consequence/transition.**

Ví dụ methodology không nhảy thẳng vào thuật toán. Nó mở bằng vấn đề cần giải quyết, rồi mới nói “To address this limitation, we introduce…”, sau đó giải thích observation làm nền cho algorithm.

Thesis mình cũng sẽ chuyển theo nhịp này.

Hiện có những đoạn kiểu:

> Composite đạt thế này. G-only thế kia. CI thế nọ. Kết luận đúng mức là... Temporal leakage thế kia...

Về khoa học không sai, nhưng nó giống **experiment report**.

Style mới sẽ là:

> Để đánh giá liệu tổ hợp composite có cải thiện chất lượng xếp hạng ngoài lợi ích về độ phủ, chúng tôi so sánh công thức đầy đủ với biến thể chỉ sử dụng căn chỉnh toàn cục. G-only đạt ... trong khi full composite đạt ...; khoảng tin cậy của chênh lệch chứa 0. Kết quả này cho thấy lợi ích chính của composite nằm ở khả năng mở rộng truy hồi, thay vì ở một tổ hợp trọng số tối ưu duy nhất.

**Thông tin y chang. Chỉ narrative đẹp hơn.**

---

## 2. Dùng transition rất nhiều, nhưng transition phải có chức năng

Paper mẫu thích những cấu trúc như:

* *To address this limitation,...*
* *In contrast,...*
* *However,...*
* *While X..., Y...*
* *This comparison demonstrates...*
* *These results indicate...*
* *Overall,...*
* *To evaluate..., we...*

Ví dụ phần LSTM mở thẳng bằng contrast với heuristic: heuristic interpretable nhưng thiếu nonlinear/long-range capability, **“To address these limitations”** mới giới thiệu LSTM.

Thesis mình cũng nên thế.

Không nên:

> Tiếp theo chúng tôi sử dụng DRfold2.

Mà:

> Mặc dù TBM cung cấp các ứng viên chất lượng cao khi có khuôn phù hợp, hiệu quả của nó suy giảm khi thư viện không chứa một cấu trúc tương đồng đủ gần. Để bổ sung các trường hợp này, chúng tôi sử dụng DRfold2 như một nguồn ứng viên độc lập...

Tức **component mới luôn xuất hiện vì component trước có một limitation cụ thể**.

---

# 3. “We” được dùng thoải mái, không cần né

Paper mẫu dùng:

> *we propose*
> *we develop*
> *we evaluate*
> *we introduce*
> *we apply*

rất tự nhiên.

T muốn thesis mình cũng bỏ bớt kiểu văn:

> “Luận văn thực hiện...”
> “Phương pháp được tiến hành...”
> “Một phép đánh giá được thực hiện...”

Nếu quy định trường cho phép, dùng:

> “Chúng tôi xây dựng...”
> “Chúng tôi đánh giá...”
> “Chúng tôi so sánh...”

sẽ giống paper khoa học hơn hẳn.

Và khi chủ thể là method:

> “Geometry v2 applies...”
> “The composite search expands...”
> “This constraint prevents...”

Không phải câu nào cũng cần “chúng tôi”.

---

# 4. Giảm mạnh chất “reviewer rebuttal”

Đây có lẽ là thay đổi lớn nhất cho thesis hiện tại.

Bản mình đang có nhiều câu kiểu:

> “Kết luận đúng mức là...”
> “Không được dùng để quy...”
> “Không component nào một mình chứng minh...”
> “Claim đúng là...”
> “Luận văn không khẳng định...”
> “Kết quả này không phải bằng chứng...”

Các câu đó **scientifically rất tốt**, nhưng lặp nhiều làm văn giống mình đang tranh luận trực tiếp với reviewer.

Paper mẫu thường truyền tải cùng một mức thận trọng bằng cấu trúc mềm hơn:

> *Although X..., Y...*
> *While X improves..., it remains...*
> *These results indicate..., rather than...*
> *This suggests that...*

Ví dụ abstract thừa nhận heuristic có accuracy cao hơn, nhưng lập tức đặt trade-off precision–recall của LSTM trong cùng một câu.

Thesis mình nên học chính cái này.

Ví dụ hiện tại:

> “Kết quả Kaggle là bằng chứng đánh giá bên ngoài cho toàn nhánh TBM, không phải bằng chứng nhân quả riêng cho từng thành phần.”

Nội dung giữ nguyên, nhưng paper-style hơn:

> “Kết quả trên tập ẩn cung cấp đánh giá bên ngoài cho toàn bộ nhánh TBM; tuy nhiên, do các thành phần không được thay đổi độc lập trong phép nộp này, mức cải thiện không thể được quy cho một bước riêng lẻ.”

Same claim. **Ít defensive hơn rất nhiều.**

---

# 5. Result paragraph sẽ theo pattern: observation → contrast → interpretation

Paper mẫu làm cái này khá consistently.

Phần qualitative:

1. Figure cho thấy gì.
2. Compare A và B.
3. Explain behavior.
4. Nêu implication.

Quantitative cũng vậy: trước tiên nêu overall ranking, sau đó contrast accuracy/F1, rồi giải thích vì sao heuristic và LSTM có behavior khác nhau theo 50/100 Mbps.

Thesis mình cũng sẽ tránh kiểu:

> A = 0.5.
> B = 0.6.
> CI = ...
> C = ...
> Kết luận...

Mà thành:

> “On the held-out set, Geometry v2 increased C1′-lDDT from ... to ..., corresponding to .... The improvement was accompanied by reductions in SW-RMSD9 and SW-RMSD15, whereas TM-score changed only marginally. Together, these results indicate that the refinement primarily affects short-range geometry while preserving the global fold.”

Đấy là đúng “paper cadence”.

---

# 6. Số liệu không đứng một mình — luôn phải có interpretation ngay sau

Paper mẫu không ném table xong để reader tự hiểu.

Sau bảng, nó lập tức nói:

> model nào cao nhất,
> trade-off nằm đâu,
> tại sao pattern như thế.

Thesis mình cũng sẽ làm vậy.

Mỗi bảng lớn cần một paragraph dưới nó trả lời:

**What is the dominant pattern?**
**What is the exception?**
**What does it imply for the RQ?**

Không cần đọc lại tất cả number trong table.

---

# 7. Word choice: thiên về động từ cụ thể

Paper mẫu viết kiểu:

* **detects**
* **distinguishes**
* **captures**
* **integrates**
* **maintains**
* **preserves**
* **reveals**
* **indicates**
* **demonstrates**
* **outperforms**
* **enables**
* **supports**

thay vì noun-heavy sentences.

Thesis mình hiện đôi lúc nhiều nominalization kiểu:

> “việc thực hiện đánh giá khả năng bổ sung của...”

Sẽ đổi thành:

> “We evaluate whether DRfold2 complements TBM...”

Hoặc tiếng Việt:

> “Chúng tôi đánh giá liệu DRfold2 có bổ sung các nếp gấp mà TBM bỏ lỡ hay không.”

Ngắn hơn và khỏe hơn.

---

# 8. Nhưng t sẽ thận trọng hơn paper mẫu ở chữ “confirm”

Paper mẫu đôi lúc dùng hơi mạnh:

> “These results confirm...”

Thesis của ní có bootstrap CI và sample nhỏ, nên t **không bê máy móc**.

T sẽ ưu tiên:

* “cho thấy”
* “chỉ ra”
* “hỗ trợ giả thuyết”
* “phù hợp với”
* “cung cấp bằng chứng cho”
* “suggests”
* “indicates”
* “supports”

và dùng “demonstrates” khi evidence thực sự mạnh.

Tức học **văn phong**, không học luôn mức độ overclaim của paper mẫu.

---

# 9. Sentence length: vừa phải, không telegram nhưng cũng không thesis-baroque

Paper mẫu thường có câu khoảng 20–35 từ, có một main clause rõ và 1 subordinate clause.

Nó không viết kiểu:

> A xảy ra. B xảy ra. C xảy ra.

nhưng cũng ít viết câu dài 5 dòng chứa 4 dấu chấm phẩy.

Thesis mình sẽ theo nhịp:

**1 câu establish.
1 câu quantify/explain.
1 câu interpret/transition.**

Khoảng **3–5 câu/paragraph** là đẹp.

---

# 10. Contrast dùng ngay trong câu

Style mẫu thích:

> **While X..., Y...**
> **Although X..., Y...**
> **In contrast,...**
> **However,...**

Cái này cực hợp với result của mình vì thesis có rất nhiều nuance:

* TBM trung bình tốt hơn DRfold2, **nhưng** DRfold2 thắng trên một số target.
* source+backbone lDDT cao hơn, **nhưng** Geometry tốt hơn SW9 và geometric diagnostics.
* Geometry improve local metrics, **nhưng** không improve TM.
* MMseqs có precision cao, **nhưng** availability thấp.

Thay vì tách thành những câu disclaimer, ta dùng **contrastive academic prose**.

---

# 11. Cách trình bày method: observation trước, formula sau

Paper mẫu không ném algorithm ngay.

Nó nói:

> capacity-limited traffic có một characteristic pattern → vì vậy density near peak có thể dùng để distinguish → rồi mới đưa 3 algorithmic steps.

Geometry v2 cũng nên vậy.

Không nên bắt đầu bằng:

[
\mathcal{L}=...
]

Mà:

> Geometry v2 is based on two complementary requirements. First, refinement should preserve the global configuration of the source structure. Second, local C1′ geometry should remain consistent with distributions observed in experimentally determined RNA structures. These requirements are encoded by the following objective...

**Rồi formula.**

Reader hiểu *why* trước *what*.

---

# 12. Formula luôn được “sandwich”

Style chuẩn mình sẽ áp dụng:

**trước equation:** equation để làm gì.

**equation.**

**sau equation:** từng term nghĩa gì và intuition.

Không để equation treo lơ lửng.

---

# 13. Lists chỉ dùng khi nó thực sự là list

Paper mẫu dùng list chủ yếu cho:

* contributions;
* sequential algorithm steps;
* explicit components.

Không biến prose thành 20 bullet points.

Thesis mình cũng vậy.

Các discussion/result nên prose nhiều hơn.

Các contribution/objective/algorithm mới dùng enumerate.

---

# 14. Figure/Table reference rất trực tiếp

Paper dùng:

> “Fig.~\ref{...} presents...”
> “Table~\ref{...} reports...”
> “As shown in Fig....”

Thesis mình sẽ thống nhất:

```latex
Fig.~\ref{fig:...}
Table~\ref{tab:...}
Section~\ref{sec:...}
```

nếu viết tiếng Anh.

Nếu thesis tiếng Việt thì có thể thống nhất:

```latex
Hình~\ref{fig:...}
Bảng~\ref{tab:...}
Mục~\ref{sec:...}
```

Không lúc “Hình”, lúc “Fig.”, lúc “hình dưới đây”.

---

# 15. Dấu và typography cũng sẽ học style đó, nhưng làm sạch hơn

T sẽ chuẩn hóa:

* `precision--recall` → output en dash “precision–recall” khi phù hợp;
* compound adjective: `time-safe`, `source-specific`, `fixed-size`;
* khoảng số: `0.02--0.08` trong LaTeX;
* đơn vị: `50~Mbps`, `0.04~\angstrom`;
* phần trăm: `95\%`;
* equation variables italic tự nhiên;
* tên method/model giữ Roman: DRfold2, Geometry v2, MMseqs2;
* không dùng quotation marks vô cớ;
* dấu colon trước explanation/list;
* semicolon dùng tiết chế, chủ yếu để nối hai independent-but-related clauses.

Và t **không copy typo** của paper mẫu như vài chỗ hyphen/en dash hoặc grammar hơi lỗi. Ta học aesthetic, không học lỗi :))

---

# 16. Heading cũng nên ngắn và descriptive hơn

Paper:

> Methodology
> Heuristic Peak Detection Algorithm
> LSTM Deep Learning Model
> Quantitative Results

Rất thẳng.

Thesis mình hiện có vài heading hơi “essay-like”:

> “Vì sao Geometry v2 cải thiện cục bộ nhưng không tăng TM-score”

Trong Discussion thì câu đó vẫn có thể hợp, nhưng overall t sẽ thiên về:

> **Local Effects of Geometry v2**

hoặc tiếng Việt:

> **Ảnh hưởng cục bộ của Geometry v2**

rồi prose bên dưới trả lời “vì sao”.

Heading không cần mang luôn conclusion nếu không cần.

---

# 17. Abstract của thesis sau này cũng nên giống nhịp paper mẫu

Target structure:

**1–2 câu:** problem/gap.

**1–2 câu:** proposed framework.

**2–4 câu:** major components và evaluation protocol.

**3–5 câu:** critical numerical findings.

**1–2 câu:** interpretation / main conclusion.

Không đưa quá nhiều methodological caveat vào abstract.

Nhưng vẫn giữ những caveat **load-bearing** như Geometry không tăng TM.

Paper mẫu làm rất tốt kiểu compress method + numeric result + nuance trong một paragraph duy nhất.

---

# 18. Conclusion cũng không recap từng experiment

Paper mẫu conclusion chỉ:

problem → proposed method → headline result → implication → future work.

Thesis dài hơn paper nên conclusion có thể chi tiết hơn, nhưng t vẫn muốn tránh:

> RQ1 có A. RQ2 có B. RQ3a có C. RQ3b có D...

theo kiểu checklist.

Thay vào đó kết luận nên đọc như **một synthesized scientific story**.

---

## Tóm lại, cái “vibe” cuối cùng t muốn cho thesis là thế này

Hiện tại prose của thesis hơi:

> **precise + defensive + experiment-log-like.**

Sau khi sửa:

> **precise + compact + narrative + evidence-driven + publication-like.**

Và nguyên tắc bất di bất dịch là:

[
\boxed{\text{Style changes; scientific content does not.}}
]

T sẽ **không** vì muốn câu đẹp hơn mà:

* đổi hypothesis;
* bỏ caveat;
* làm CI chứa 0 nghe như significant;
* đổi interpretation;
* nâng empirical contribution thành methodological novelty;
* làm Geometry nghe như tăng TM;
* làm DRfold2 nghe như vượt TBM;
* giấu negative result.

Chỉ thay **cách kể**.

Một ví dụ rất điển hình là đoạn thesis hiện tại đang viết khá “rebuttal-like”: nó mô tả Kaggle improvement rồi lập tức nhấn mạnh đây không phải causal evidence cho từng component.  Sau này nội dung đó vẫn nguyên xi về ý nghĩa, nhưng sẽ được viết thành một paragraph flow tự nhiên kiểu **result → qualification → implication**, giống cách paper mẫu xử lý trade-off giữa các model.

Vậy từ giờ khi ní bảo t **“sửa văn thesis”**, t sẽ mặc định theo style profile này. Riêng preference này chưa được lưu thành memory vì phiên này không cho cập nhật memory sau khi đã dùng file tools; nếu ní muốn nó được giữ xuyên các chat mới thì nhắc t ở một conversation mới là được.
