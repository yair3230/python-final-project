  > 📝 **הערה:** יש עוד עבודה לסדר את הקבצים ובעיקר לאחד את הוויזואליזציות לgraphs.py ואז יהיה אפשר לערוך את הmain שיהיה קריאות מינימליות לפונקציות של הגרפים. בנוסף מוזמנים לעבור על הREADME ולהציע שינויים בתוכן/מבנה שלו בהתאם להבנתכם את הפרוייקט . 
---

# Neuro-Data Analysis: Socio-Educational Background vs. Reasoning & Math Skills

---

## 👥 Presenters

* **Yair Olmayev**
* **Roy Carmelli**
* **Ben Kopler**
* **Matan Spiegel**

---

## 🔬 General Research Overview

This project investigates how a child’s **socio-educational environment** (e.g., parental background and school context) relates to **cognitive/academic outcomes**, with an emphasis on **mathematical achievement** and **deductive reasoning** abilities.

Unlike prior work on this dataset that focused mainly on neural (fMRI) correlates of reasoning, our goal is to examine the **behavioral + background** perspective. We aim to quantify how much *environment* adds to the prediction of ability beyond core cognitive baselines.

We work with the OpenNeuro dataset **ds002886** (*Brain Development and Reasoning*).
The analysis is based on a merged subject-level table (`main_dataset.csv`) together with trial-level behavioral logs (`trial_data/`), used to compute measures, run EDA, and fit statistical models.

### 📚 Dataset References

* **OpenNeuro dataset:** ds002886 (v1.1.0)
* **Repo subject table:** `main_dataset.csv`
* **Repo trial logs:** `trial_data/`

---

## ❓ The Research Questions We Raised

1. **Association Question (Correlational)**
How is the socio-educational context (including socio-economic background, school characteristics, and parental professional background) related to mathematical performance and deductive reasoning abilities?
2. **Incremental Contribution Question (Modeling)**
Does this relationship remain significant after controlling for individual cognitive variables, specifically IQ components (VIQ, PIQ, FSIQ) and working memory?
📝 הערה: אמור להיות כחלק מהEDA .
3. **Interpretation Question (Insights)**
What can we conclude about the relative weight of *environment* (Nurture) in the development of different thinking skills (mathematics vs. logic/reasoning)?

---

## 📂 The Data and Its Structure

This repository contains two main data layers:

### 1️⃣ Subject-Level Table

* **File:** `main_dataset.csv`
* **Unit:** One row per subject (56 subjects total)
* **Contents:** Background variables + cognitive/achievement summaries
(see `data_description.json`)

### 2️⃣ Trial-Level Behavioral Logs

* **Folder:** `trial_data/`
* **Unit:** Data per trial, per child
* **Use:** Logs are processed to derive behavioral reasoning metrics aggregated to subject level

### 🔗 Extra Reference Mapping

* **File:** `isco.csv`
Used as a reference table for mapping occupational categories (ISCO standards) during feature engineering.

---

## 🧱 Project Structure

The repository follows a modular structure where **logic**, **analysis**, and **visualization** are separated.

* `main.ipynb`
Main end-to-end notebook.
Orchestrates the analysis and **calls visualization functions from `graphs.py**`, containing minimal logic.
> 📝 **הערה:** כרגע `main.ipynb` מכיל קוד גרפים פנימי או קורא למודולים אחרים.
> יש לבצע רפקטור כך שכל הוויזואליזציה תיקרא אך ורק דרך פונקציות מתוך `graphs.py`.


* `graphs.py`
**Visualization module.**
Contains all plotting functions (scatter plots, regression lines, correlation matrices).
> 📝 **הערה:** קובץ זה אינו קיים עדיין ברפוזיטורי.
> יש ליצור אותו ולהעביר אליו את כל פונקציות הציור מהמחברות או מ־`data_analysis.py`.


* `EDA/` – Exploratory analysis artifacts and helper notebooks
* `eda_background_math_verbal.ipynb` – Focused EDA: background vs. math/verbal measures
* `check.ipynb` – Sanity checks and quick validation
* `data_merger.py` – Data integration pipeline (trial logs → subject-level metrics)
* `data_analysis.py` – Statistical tests and hierarchical regression modeling
* `data_tools.py` – Shared utilities (loading, cleaning, aggregations, transformations)
* `consts.py` – Project constants (paths, column names, configuration flags)
* `data_description.json` – Data dictionary for `main_dataset.csv`
* `data_tests/` – Tests validating major pipeline stages
* `requirements.txt` – Python dependencies
* `README.md` – Project documentation

---

## ⚙️ How We Process the Data in Code to Answer the Questions

We follow a standard **“DA with Python”** pipeline implemented as modular building blocks:

### Stage A — Data Loading & Validation

* Load `main_dataset.csv` and trial logs from `trial_data/`
* Run integrity checks.
* **Data Selection:** Due to high attrition (~19% missing in Session 2), **we strictly use Session 1 data** for the cross-sectional analysis.
* **Code locations:** `data_tools.py`, `data_tests/`, `check.ipynb`

### Stage B — Feature Engineering (Data Engineering)

* Extract reasoning metrics from behavioral logs
* Merge engineered features into the main table
* **Code locations:** `data_merger.py`

### Stage C — Exploratory Data Analysis (EDA)

* Analyze distributions and correlations between background variables and cognitive scores
* **Visualization:** All EDA plots are generated by calling functions from `graphs.py`
> 📝 **הערה:** כיום קוד הוויזואליזציה מפוזר במחברות או ב־`main.ipynb`.
> יש לאחד את כל הוויזואליזציה כך שתגיע מ־`graphs.py` בלבד.


* **Hypothesis check:**
* Humanities-oriented parents → stronger verbal skills
* Real/STEM-oriented parents → stronger mathematical skills


* **Code locations:** `eda_background_math_verbal.ipynb`, `graphs.py`

### Stage D — Statistical Modeling (Hierarchical Regression)

1. **Baseline Model:** IQ components (VIQ, PIQ, FSIQ) and working memory
2. **Extended Model:** Add socio-educational predictors and test change in explained variance

* **Code locations:** `data_analysis.py`, `main.ipynb`

### Stage E — Results & Interpretation

* Report effect directions, significance, and conclusions regarding the *Nurture* contribution
* **Visualization:** Final regression plots and summary charts are drawn using `graphs.py`
> 📝 **הערה:** יש להעביר גם את גרפי הרגרסיה הסופיים לפונקציות ב־`graphs.py`.



---

## 🏆 Success Definition and Criteria

1. **Data Quality Success**
Final merged table includes at least **80% of subjects**
2. **Statistical Success**
A **statistically valid** regression model that meets standard assumptions (e.g., normality of residuals), allowing us to confirm or refute the hypothesis regarding the environmental contribution.
3. **Insight Success**
Clear, data-backed conclusions regarding the role of *environment* in developing different thinking skills

---

## 💻 How to Work with the Code

### 1️⃣ Environment Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

```

### 2️⃣ Recommended Run Path (Notebooks)

1. **Open `main.ipynb**`
2. Run cells top-to-bottom.

> 📝 **הערה:** `main.ipynb` אמור לייבא את `graphs` ו-`data_analysis` ולהכיל מינימום לוגיקה פנימית.
> לאחר יצירת `graphs.py`, יש לוודא שה-imports עובדים כראוי.

**Optional supporting notebooks:**

* `eda_background_math_verbal.ipynb` – focused EDA
* `check.ipynb` – sanity checks

### 3️⃣ Tests

To run the project tests:

```bash
pytest -q

```





