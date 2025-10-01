# From Silent Goodbyes to Clear Signals: A Story behind a Trustworthy Model

Imagine we’re detectives in a bank, combing through thousands of customer breadcrumbs to spot who’s quietly getting ready to leave. Our journey begins with the unglamorous tidying work-IDs, taming wild numbers, and sculpting features that actually mean something, and then moves through honest visuals that reveal the shape of the problem. Armed with a toolkit ranging from straightforward Logistic Regression and finding the patterns for SVM sensitiviy to rugged Random Forests and nimble XGBoost, we don’t stop at AUC bragging rights. We demand reasons. With DALEX, ROC curves, and Youden’s J, we uncover the reasons behind each prediction and where to set decision thresholds. This is the full story: clean data, leak-proof preprocessing, fair model bake-offs, smart tuning, and clear interpretation, hence the model is not only just accurate, but also trustworthy, but still actionable yet managable for keeping customers before they give goodbyes.

# Understanding Our Generals

This notebook works like a small investigation file: each row is a customer snapshot: who they are, which bank they trusted the most, what products they hold, and whether they left. Before any modeling or visualization, we profile, clean, and enrich the fields below so downstream findings are trustworthy, reproducible, and actionable.

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
| `HasCrCard`      | `bool`      | Owns a credit card (0/1)           | Map `{0,1}→{False,True}`; ensure boolean dtype |
| `IsActiveMember` | `bool`      | Active membership flag (0/1)       | Map to boolean; key engagement signal |
| `EstimatedSalary`| `float`     | Estimated annual salary            | Check scale/units; use in ratios (`Balance_to_Salary`) |
| `Exited`         | `int (0/1)` | Churn label (1 = churned)          | **Target variable**; exclude from features; use for stratified splits & evaluation |
