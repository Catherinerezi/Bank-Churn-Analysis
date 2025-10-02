# From Silent Goodbyes to Clear Signals: A Story behind a Trustworthy Model

Imagine we’re detectives in a bank, combing through thousands of customer breadcrumbs to spot who’s quietly getting ready to leave. Our journey begins with the unglamorous tidying work-IDs, taming wild numbers, and sculpting features that actually mean something, and then moves through honest visuals that reveal the shape of the problem. Armed with a toolkit ranging from straightforward Logistic Regression and finding the patterns for SVM sensitiviy to rugged Random Forests and nimble XGBoost, we don’t stop at AUC bragging rights. We demand reasons. With DALEX, ROC curves, and Youden’s J, we uncover the reasons behind each prediction and where to set decision thresholds. This is the full story: clean data, leak-proof preprocessing, fair model bake-offs, smart tuning, and clear interpretation, hence the model is not only just accurate, but also trustworthy, but still actionable yet managable for keeping customers before they give goodbyes.

# Understanding Our Generals

This notebook works like a small investigation file: each row is a customer snapshot: who they are, which bank they trusted the most, what products they hold, and whether they left or stay. Before any modeling or visualization, we profile, clean, and enrich the fields below so downstream findings are trustworthy, reproducible, and actionable.

| Column           | Type        | Meaning                            | Notes & Handling |
|------------------|-------------|------------------------------------|------------------|
| `RowNumber`      | `str`       | Row index from the source file     | Identifier only; keep for trace/debug; **drop before modeling** |
| `CustomerId`     | `str`       | Unique customer identifier         | Use for joins/dedup checks; **drop before modeling** |
| `Surname`        | `category`  | Customer last name                 | Personally identifying text; optional for QA; **drop before modeling** |
| `CreditScore`    | `int`       | Credit score                       | Coerce numeric; sanity-check range; consider buckets/flags (`LowCreditScore`, `HighCreditScore`) |
| `Geography`      | `category`  | Customer country/region            | Cast to category; one-hot encode; group rare levels to “Other” if sparse |
| `Gender`         | `category`  | Customer gender                    | Cast to category; one-hot encode; treat as **sensitive**—review policy before production use |
| `Age`            | `int`       | Customer age                       | Check outliers; engineer `Age_Bin`, `IsSenior_55plus`; interaction `Age_x_Active` |
| `Tenure`         | `int`       | Years with the bank                | Keep integer; derive `Tenure_per_Age` for loyalty context |
| `Balance`        | `float`     | Account balance                    | Skewed; consider winsorizing/clamping; create `HighBalance`; ratio `Balance_to_Salary` |
| `NumOfProducts`  | `int`       | Number of bank products held       | Discrete count; consider interaction `Products_x_Active` |
| `HasCrCard`      | `bool`      | Owns a credit card (0/1)           | Map `{0,1}` to `{False,True}`; ensure boolean dtype |
| `IsActiveMember` | `bool`      | Active membership flag (0/1)       | Map to boolean; key engagement signal |
| `EstimatedSalary`| `float`     | Estimated annual salary            | Check scale/units; use in ratios (`Balance_to_Salary`) |
| `Exited`         | `int (0/1)` | Churn label (1 = churned)          | **Target variable**; exclude from features; use for stratified splits & evaluation |

**General quality checks**
- Fix identifiers:
  - Cast to `str`
  - categoricals to `category`
  - binary fields to `bool`
- Impute:
  - median for numeric
  - `"Unknown"` for categorical
  - one-hot encode with `handle_unknown="ignore"`
- Prevent leakage:
  - fit preprocessing on **train only**
  - drop ID/text columns before modeling
- Quality checks:
  - missing-value audit
  - type coercion
  - outlier review
  - class-balance review (use stratification)

# Attachments
- [Data Processing](https://colab.research.google.com/drive/1YfTZZpgwdG_dvaVgbFUF0JDxaaQRJXm8#scrollTo=6O3l7ZT0wMZd)

# What we bring to the table?

## Why the model is useful (via Cumulative Gain)?

**Cumulative Gain shows** how many churners we can catch if we only contact the highest-risk customers first. Our curve climbs far above the random baseline, meaning a small, focused campaign can recover most churn risk without messaging everyone.

We built with a leak-proof pipeline (fit on train only, stratified split, one hot + scaling), then computed gains on the held-out test set to make the curve reflects true generalization.

<p align="center">
  <img src="https://github.com/Catherinerezi/Bank-Churn-Analysis/blob/main/assets/CumulativeGain.png" alt="Cumulative Gain: model vs baseline" width="560">
</p>

**How to read it?**
- Baseline (diagonal line): contacting k% of customers at random catches k% of churners.
- Model curve (ours): the higher and steeper it is above the baseline, especially in the first 10–30%, the more efficient our targeting.
- Elbow point: where the curve starts flattening, we got this in diminishing returns; past this, extra budget catch smaller gains.
- Lift at k%: gain divided by k. A lift of 3 at 20%, means we’re 3× better than random in that slice.

**What we observed?**

<p align="center">
  <img src="https://github.com/Catherinerezi/Bank-Churn-Analysis/blob/main/assets/Lift%20Curve.png" alt="Cumulative Gain: model vs baseline" width="560">
</p>

| % Sampel | N Target | Churn Catch | Cumulative Gain | Lift |
|---------:|---------:|------------:|----------------:|-----:|
| 10       | 200      | 158         | 0.388           | 3.88 |
| 20       | 400      | 245         | 0.602           | 3.01 |
| 30       | 601      | 301         | 0.740           | 2.47 |
| 40       | 800      | 334         | 0.821           | 2.05 |
| 50       | 1000     | 356         | 0.875           | 1.75 |
| 60       | 1200     | 373         | 0.916           | 1.53 |
| 70       | 1401     | 391         | 0.961           | 1.37 |
| 80       | 1600     | 400         | 0.983           | 1.23 |
| 90       | 1800     | 402         | 0.988           | 1.10 |
| 100      | 2000     | 407         | 1.000           | 1.00 |

- Top **10%** captures **~XX%** of churn (lift about **L1×**).  
- Top **20%** captures **~YY%** of churn (lift about **L2×**).  
- Gains **flatten after ~KK%** coverage; by **50%**, we’ve already caught **~ZZ%** of churn.

**Why it matters:**
- Target size: choose the smallest % that hits your capture goal (e.g., 20% list  caught about 60% churn).
- Budget & ops: fewer messages, lower spend, less customer fatigue.
- Policy: start with top-k targeting (e.g., top 20%) for campaigns, keep ROC/Youden for thresholding when you need hard “churn / not churn” decisions in a product.

## How big is the problem?

**Churn Distribution** shows how many customers left vs stayed. This baseline matters because it sets the difficulty of the task: if churn is a minority class, naive accuracy can look high while missing most churners. We use this view to size the problem before modeling.

Countplot built from the training split to prevent peeking (labels come from `y_train` in the notebook).

<p align="center">
  <img src="https://github.com/Catherinerezi/Bank-Churn-Analysis/blob/main/assets/ChurnDistribution.png" alt="Stayed vs Churned distribution" width="420">
</p>

**How to read the chart (signals)?**
- Bars (Stayed vs Churned): height = count (or percent).
- Imbalance cue: a much shorter “Churn” bar means fewer positives, means harder recall.

**What we observed (training set)**
- **Churn rate:** ~XX%  
- **Stay rate:** ~(100–XX)%  
- Pattern indicates **[moderate/strong] imbalance**, so we use class weights and calibrated thresholds.

**Why it matters?**
- We stratify train/test splits to keep this ratio consistent.
- We rely on ROC-AUC instead of raw accuracy.
- We consider class weights and threshold tuning (Youden’s J) so the model doesn’t miss churners.
- For business ops, the base rate anchors expected campaign volume (e.g., if churn about 20%, "catching half" means about 10% of customers).

**Where the churn concentrates (via Number of Products)?**

**This barchart below** asks a simple question: *does holding more (or fewer) products relate to churn?*  
It’s a practical lens because it points straight at segments we can act on—bundling, cross-sell, or retention offers.

<sub>Built from the training split (no leakage). Encoding and preprocessing follow the same pipeline used in modeling.</sub>
