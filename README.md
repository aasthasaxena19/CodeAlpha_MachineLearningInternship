# 💳 Credit Scoring Model — CodeAlpha Internship Task 1

A Machine Learning model to predict whether a loan applicant will **repay** or **default** on their loan, based on their financial history.

---

## 🎯 Objective
Predict an individual's creditworthiness using past financial data with classification algorithms.

## 📁 Project Structure
```
CodeAlpha_CreditScoring/
│
├── credit_scoring.py          ← Main Python script (heavily commented)
├── credit_scoring_results.png ← Auto-generated charts
├── requirements.txt           ← Libraries needed
└── README.md                  ← This file
```

## 🧠 What This Model Does

| Step | Description |
|------|-------------|
| 1️⃣ | Creates a realistic dataset of 1000 loan applicants |
| 2️⃣ | Explores and visualizes the data (EDA) |
| 3️⃣ | Preprocesses data (encoding, scaling, feature engineering) |
| 4️⃣ | Trains **Logistic Regression** and **Random Forest** models |
| 5️⃣ | Evaluates using Accuracy, Precision, Recall, F1, ROC-AUC |
| 6️⃣ | Generates 6 visualization charts |
| 7️⃣ | Predicts for new applicants |

## 📊 Features Used
| Feature | Description |
|---------|-------------|
| Age | Applicant's age |
| Income | Annual income |
| Loan Amount | Amount borrowed |
| Credit Score | 300–850 range |
| Late Payments | Number of late payments |
| Debt-to-Income Ratio | Total debt / Income |
| Employment Years | Work experience |
| Education | Highest degree |
| + More... | |

## 🚀 How to Run

### Step 1: Install requirements
```bash
pip install -r requirements.txt
```

### Step 2: Run the model
```bash
python credit_scoring.py
```

### Step 3: View Results
- Results printed in the terminal
- Charts saved as `credit_scoring_results.png`

## 📈 Model Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~75% | ~0.82 |
| Random Forest | ~82% | ~0.89 |

**🏆 Winner: Random Forest** — handles complex patterns better!

## 📚 Key Concepts (For Beginners)

- **Accuracy** → What % of predictions were correct?
- **Precision** → Of all "Default" predictions, how many were actually defaults?
- **Recall** → Of all actual defaults, how many did the model catch?
- **F1-Score** → Balance between Precision and Recall
- **ROC-AUC** → How well the model separates good/bad customers (1.0 = perfect)
- **Confusion Matrix** → Table showing correct vs wrong predictions

## 🛠️ Libraries Used
- `pandas` — Data manipulation
- `numpy` — Math operations
- `scikit-learn` — Machine learning models
- `matplotlib` & `seaborn` — Data visualization

## 👤 Author
CodeAlpha Machine Learning Internship
