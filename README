# Neuro-Data Analysis: Socio-Educational Background vs. Reasoning & Math Skills

---

## 👥 Presenters
- **Yair Olmayev** (314620535)
- **Roy Carmelli** (323834515)
- **Ben Kopler** (204085062)
- **Matan Spiegel** (207014648)

---

## 🔬 General Research Overview
This project investigates how a child’s **socio-educational environment** (e.g., parental background and school context) relates to **cognitive and academic outcomes**, with a focus on **mathematical achievement** and **deductive reasoning**.

Unlike prior work on this dataset that focused mainly on neural (fMRI) correlates of reasoning, this project adopts a **behavioral and socio-educational perspective**.  
Our goal is to quantify how much **environmental factors** contribute to performance **beyond core cognitive baselines**.

We use the OpenNeuro dataset **ds002886** (*Brain Development and Reasoning*), combining a merged subject-level table (`main_dataset.csv`) with trial-level behavioral logs (`trial_data/`) for feature extraction, EDA, and statistical modeling.

### 📚 Dataset References
- **OpenNeuro dataset:** ds002886 (v1.1.0)
- **Subject-level table:** `main_dataset.csv`
- **Trial-level logs:** `trial_data/`

---

## ❓ Research Questions
1. **Association (Correlational)**  
   How is socio-educational context (socio-economic background, school characteristics, parental professional background) related to **mathematical performance** and **deductive reasoning abilities**?

2. **Incremental Contribution (Modeling)**  
   Does this relationship remain significant **after controlling for individual cognitive variables**, specifically IQ components (VIQ, PIQ, FSIQ) and working memory?

3. **Interpretation (Insights)**  
   What can be concluded about the relative weight of **environment (“nurture”)** in the development of different thinking skills (mathematics vs. logic/reasoning)?

---

## 📂 Data and Structure

### 1️⃣ Subject-Level Table
- **File:** `main_dataset.csv`
- **Unit:** One row per subject (56 subjects total)
- **Contents:** Socio-demographic background + cognitive/achievement summaries  
  (see `data_description.json`)

### 2️⃣ Trial-Level Behavioral Logs
- **Folder:** `trial_data/`
- **Unit:** Per trial, per child
- **Purpose:** Used to derive behavioral reasoning metrics aggregated to subject level

### 🔗 Reference Mapping
- **File:** `isco.csv`  
  Used for mapping parental occupations (ISCO standard) during feature engineering.

---

## 🧱 Project Structure
The repository is organized to clearly separate **logic**, **analysis**, and **visualization**.

- `main.ipynb`  
  Main orchestration notebook.  
  Responsible for running the pipeline and **calling plotting functions** (minimal logic).

  > **TODO:** Refactor so that all visualization code is called exclusively via `graphs.py`.

- `graphs.py`  
  **Visualization module**.  
  Contains all plotting utilities (scatter plots, regression lines, correlation matrices).

  > **TODO:** Create this module and move all plotting functions from notebooks / `data_analysis.py` into it.

- `EDA/` – Exploratory analysis artifacts  
- `eda_background_math_verbal.ipynb` – Focused EDA: background vs. math/verbal abilities  
- `check.ipynb` – Sanity checks and quick validation  
- `data_merger.py` – Data integration and feature aggregation  
- `data_analysis.py` – Statistical analysis and hierarchical regression modeling  
- `data_tools.py` – Shared utilities (loading, cleaning, transformations)  
- `consts.py` – Centralized constants and configuration  
- `data_description.json` – Dataset dictionary  
- `data_tests/` – Tests validating major pipeline stages  
- `requirements.txt` – Python dependencies  
- `README.md` – Project documentation  

---

## ⚙️ Data Processing & Analysis Pipeline

### Stage A — Data Loading & Validation
- Load subject and trial-level data
- Handle missingness (including ~19% missing Session 2)
- **Code:** `data_tools.py`, `data_tests/`, `check.ipynb`

### Stage B — Feature Engineering
- Extract reasoning metrics from behavioral logs
- Aggregate to subject-level features
- Merge into main dataset
- **Code:** `data_merger.py`

### Stage C — Exploratory Data Analysis (EDA)
- Analyze distributions and correlations between background and cognitive outcomes
- **Visualization:** Generated exclusively via `graphs.py`

  > **TODO:** Ensure all EDA visualizations are imported from `graphs.py`.

- **Hypothesis:**  
  - Humanities-oriented parental background → stronger verbal abilities  
  - Realistic/STEM parental background → stronger mathematical abilities

- **Code:** `eda_background_math_verbal.ipynb`, `graphs.py`

### Stage D — Statistical Modeling (Hierarchical Regression)
1. **Baseline Model:** IQ components (VIQ, PIQ, FSIQ) + working memory  
2. **Extended Model:** Add socio-educational predictors and test ΔR²

- **Code:** `data_analysis.py`, `main.ipynb`

### Stage E — Results & Interpretation
- Report effect sizes, significance, and conclusions about environmental contribution
- **Visualization:** Final plots drawn using `graphs.py`

  > **TODO:** Move final regression plots to `graphs.py`.

---

## 🏆 Success Criteria
1. **Data Quality:** ≥ 80% of subjects included in final merged dataset  
2. **Statistical Validity:** Interpretable hierarchical regression addressing the research question  
3. **Insight:** Clear, data-supported conclusions on environmental influence

---

## 💻 How to Work with the Code

### Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
Running the Project
Open main.ipynb

Run cells top-to-bottom

Note: main.ipynb imports data_analysis and graphs and avoids heavy logic blocks.

TODO: Verify imports after creating graphs.py.

Tests
bash
Copy code
pytest -q
