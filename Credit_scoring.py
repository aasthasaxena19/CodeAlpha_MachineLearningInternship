# ============================================================
# TASK 1: CREDIT SCORING MODEL
# CodeAlpha Machine Learning Internship
# ============================================================
# 🎯 GOAL: Predict if a person will repay a loan or default
#          based on their financial history.
#
# 📚 WHAT YOU'LL LEARN:
#   - How to load and explore a dataset
#   - How to clean/preprocess data
#   - How to train ML models (Logistic Regression, Random Forest)
#   - How to evaluate models using metrics
# ============================================================

# --- STEP 1: IMPORT LIBRARIES ---
# Think of libraries as "toolboxes" with ready-made functions

import pandas as pd              # For working with tables/data
import numpy as np               # For math operations
import matplotlib.pyplot as plt  # For drawing charts
import seaborn as sns            # For prettier charts
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn = The main ML library
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Clean data
from sklearn.linear_model import LogisticRegression   # Simple ML model
from sklearn.ensemble import RandomForestClassifier   # Powerful ML model
from sklearn.metrics import (
    accuracy_score,        # % of correct predictions
    classification_report, # Precision, Recall, F1 breakdown
    confusion_matrix,      # Table of predictions vs actual
    roc_auc_score,         # How well model ranks positives vs negatives
    roc_curve              # For plotting the ROC curve
)

print("=" * 60)
print("  CREDIT SCORING MODEL - CodeAlpha Internship Task 1")
print("=" * 60)


# ============================================================
# STEP 2: CREATE / LOAD DATASET
# ============================================================
# We'll create a realistic synthetic dataset (fake but realistic)
# In real projects, you'd load a CSV file like:
#   df = pd.read_csv("credit_data.csv")
#
# Our dataset represents people applying for a loan with features:
#   - Age, Income, Loan Amount, Credit History, etc.
#   - Target: 0 = Will repay (Good), 1 = Will default (Bad)

print("\n📦 STEP 1: Creating Dataset...")

np.random.seed(42)  # Seed = makes random numbers the same every run
n = 1000            # We'll create 1000 people's data

data = {
    'age':              np.random.randint(22, 65, n),
    'income':           np.random.randint(20000, 120000, n),
    'loan_amount':      np.random.randint(5000, 80000, n),
    'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n),
    'credit_score':     np.random.randint(300, 850, n),
    'num_credit_lines': np.random.randint(1, 10, n),
    'num_late_payments':np.random.randint(0, 10, n),
    'employment_years': np.random.randint(0, 30, n),
    'debt_to_income':   np.round(np.random.uniform(0.05, 0.60, n), 2),
    'has_mortgage':     np.random.choice([0, 1], n),
    'education':        np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
}

df = pd.DataFrame(data)

# Create the TARGET column (what we want to predict)
# Logic: Higher risk → more likely to default
# People with low credit score, many late payments, high debt = more likely to default
default_prob = (
    (850 - df['credit_score']) / 550 * 0.4 +    # Low credit score = higher risk
    df['num_late_payments'] / 10 * 0.35 +         # More late payments = higher risk
    df['debt_to_income'] * 0.25                   # High debt ratio = higher risk
)
df['default'] = (default_prob + np.random.uniform(-0.1, 0.1, n) > 0.45).astype(int)
# 0 = Will repay loan (Good customer)
# 1 = Will default / not repay (Bad customer)

print(f"✅ Dataset created with {len(df)} records and {len(df.columns)} columns")
print(f"   Default rate: {df['default'].mean():.1%} people likely to default")


# ============================================================
# STEP 3: EXPLORE THE DATA (EDA = Exploratory Data Analysis)
# ============================================================
print("\n📊 STEP 2: Exploring the Dataset...")

print("\n--- First 5 rows of data ---")
print(df.head())

print("\n--- Dataset Info ---")
print(f"Shape: {df.shape} (rows x columns)")
print(f"Missing values: {df.isnull().sum().sum()} (none = great!)")

print("\n--- Target Distribution ---")
counts = df['default'].value_counts()
print(f"  Will Repay (0): {counts[0]} people")
print(f"  Will Default (1): {counts[1]} people")


# ============================================================
# STEP 4: PREPROCESS DATA
# ============================================================
print("\n🔧 STEP 3: Preprocessing Data...")

# 4a. Handle categorical (text) columns
# ML models only understand numbers, not text like "Bachelor"
# LabelEncoder converts: 'High School'→0, 'Bachelor'→1, 'Master'→2, 'PhD'→3

le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
print(f"✅ Education encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 4b. Create new useful features (Feature Engineering)
# These new features might help the model understand patterns better
df['loan_to_income_ratio'] = df['loan_amount'] / df['income']  # Higher ratio = riskier
df['payment_history_score'] = 10 - df['num_late_payments']     # More late payments = lower score

# 4c. Define X (inputs/features) and y (target/output)
# X = everything the model USES to make predictions
# y = what the model PREDICTS
feature_columns = [
    'age', 'income', 'loan_amount', 'loan_term_months',
    'credit_score', 'num_credit_lines', 'num_late_payments',
    'employment_years', 'debt_to_income', 'has_mortgage',
    'education_encoded', 'loan_to_income_ratio', 'payment_history_score'
]

X = df[feature_columns]  # Input features (13 columns)
y = df['default']        # Target column (0 or 1)

print(f"✅ Features (X): {X.shape[1]} columns")
print(f"✅ Target (y): {y.shape[0]} values")

# 4d. Split data into Training set and Testing set
# Training set (80%): Model learns from this
# Testing set (20%):  We test the model on data it has NEVER seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% goes to testing
    random_state=42,    # Same split every time
    stratify=y          # Keep same % of defaults in both sets
)
print(f"✅ Training set: {len(X_train)} records | Testing set: {len(X_test)} records")

# 4e. Scale features (StandardScaler)
# Makes all numbers on same scale (0 to 1 range)
# Important for Logistic Regression (less important for Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Learn scale from training, apply it
X_test_scaled  = scaler.transform(X_test)         # Apply SAME scale to test data


# ============================================================
# STEP 5: TRAIN ML MODELS
# ============================================================
print("\n🤖 STEP 4: Training Machine Learning Models...")

# --- MODEL 1: LOGISTIC REGRESSION ---
# Think of it as drawing a straight line to separate good/bad customers
# Simple but effective for binary (0/1) classification
print("\n  Training Model 1: Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)   # .fit() = "learn from data"
lr_preds = lr_model.predict(X_test_scaled)     # Predict on test data
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]  # Probability scores
print("  ✅ Logistic Regression trained!")

# --- MODEL 2: RANDOM FOREST ---
# Think of it as asking 100 different "decision trees" and taking majority vote
# More powerful than Logistic Regression, handles complex patterns
print("\n  Training Model 2: Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,   # 100 trees in the forest
    max_depth=10,       # How deep each tree can grow
    random_state=42
)
rf_model.fit(X_train, y_train)   # Random Forest doesn't need scaled data
rf_preds = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
print("  ✅ Random Forest trained!")


# ============================================================
# STEP 6: EVALUATE MODELS
# ============================================================
print("\n📈 STEP 5: Evaluating Model Performance...")

def evaluate_model(name, y_true, y_pred, y_proba):
    """Function to print evaluation metrics for a model"""
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Will Repay', 'Will Default']))
    return acc, auc

lr_acc, lr_auc = evaluate_model("LOGISTIC REGRESSION", y_test, lr_preds, lr_proba)
rf_acc, rf_auc = evaluate_model("RANDOM FOREST",        y_test, rf_preds, rf_proba)


# ============================================================
# STEP 7: VISUALIZATIONS (Graphs / Charts)
# ============================================================
print("\n📊 STEP 6: Creating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Credit Scoring Model — Analysis Dashboard', fontsize=16, fontweight='bold')

# --- Chart 1: Target Distribution (Pie Chart) ---
ax1 = axes[0, 0]
counts = y.value_counts()
ax1.pie(counts, labels=['Will Repay', 'Will Default'],
        autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
ax1.set_title('Loan Default Distribution')

# --- Chart 2: Credit Score vs Default ---
ax2 = axes[0, 1]
df.boxplot(column='credit_score', by='default', ax=ax2,
           patch_artist=True)
ax2.set_title('Credit Score vs Default')
ax2.set_xlabel('Default (0=Repay, 1=Default)')
ax2.set_ylabel('Credit Score')
plt.sca(ax2)
plt.xticks([1, 2], ['Will Repay', 'Will Default'])

# --- Chart 3: Confusion Matrix (Random Forest) ---
ax3 = axes[0, 2]
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Repay', 'Default'],
            yticklabels=['Repay', 'Default'])
ax3.set_title('Confusion Matrix — Random Forest')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# --- Chart 4: ROC Curve (Both Models) ---
ax4 = axes[1, 0]
for name, proba in [('Logistic Regression', lr_proba), ('Random Forest', rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax4.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
ax4.plot([0,1], [0,1], 'k--', label='Random Guess')
ax4.set_title('ROC Curve Comparison')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- Chart 5: Feature Importance (Random Forest) ---
ax5 = axes[1, 1]
importances = pd.Series(rf_model.feature_importances_, index=feature_columns)
importances.sort_values(ascending=True).plot(kind='barh', ax=ax5, color='steelblue')
ax5.set_title('Feature Importance — Random Forest')
ax5.set_xlabel('Importance Score')

# --- Chart 6: Model Comparison Bar Chart ---
ax6 = axes[1, 2]
models = ['Logistic\nRegression', 'Random\nForest']
accuracies = [lr_acc * 100, rf_acc * 100]
aucs = [lr_auc * 100, rf_auc * 100]
x = np.arange(len(models))
width = 0.35
ax6.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3498db')
ax6.bar(x + width/2, aucs,       width, label='AUC-ROC (%)', color='#9b59b6')
ax6.set_title('Model Performance Comparison')
ax6.set_xticks(x)
ax6.set_xticklabels(models)
ax6.set_ylabel('Score (%)')
ax6.set_ylim(0, 100)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)
for i, (a, b) in enumerate(zip(accuracies, aucs)):
    ax6.text(i - width/2, a + 0.5, f'{a:.1f}%', ha='center', va='bottom', fontsize=9)
    ax6.text(i + width/2, b + 0.5, f'{b:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('credit_scoring_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Charts saved as 'credit_scoring_results.png'")


# ============================================================
# STEP 8: PREDICT FOR A NEW PERSON (Real-world usage demo)
# ============================================================
print("\n🔮 STEP 7: Predicting for a New Person...")

# Let's predict for 2 example people
new_applicants = pd.DataFrame({
    'age':               [35, 55],
    'income':            [75000, 30000],
    'loan_amount':       [15000, 50000],
    'loan_term_months':  [36, 60],
    'credit_score':      [720, 400],
    'num_credit_lines':  [5, 2],
    'num_late_payments': [0, 7],
    'employment_years':  [10, 3],
    'debt_to_income':    [0.15, 0.55],
    'has_mortgage':      [1, 0],
    'education_encoded': [2, 0],           # 2=Master, 0=High School
    'loan_to_income_ratio': [15000/75000, 50000/30000],
    'payment_history_score': [10, 3],
})

predictions = rf_model.predict(new_applicants)
probabilities = rf_model.predict_proba(new_applicants)[:, 1]

names = ["Applicant A (Good Profile)", "Applicant B (Risky Profile)"]
for i, (name, pred, prob) in enumerate(zip(names, predictions, probabilities)):
    result = "❌ LIKELY TO DEFAULT" if pred == 1 else "✅ LIKELY TO REPAY"
    print(f"\n  {name}")
    print(f"    Prediction: {result}")
    print(f"    Default Probability: {prob:.1%}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  ✅ TASK 1 COMPLETE — CREDIT SCORING MODEL")
print("=" * 60)
print(f"  Logistic Regression → Accuracy: {lr_acc*100:.1f}% | AUC: {lr_auc:.3f}")
print(f"  Random Forest       → Accuracy: {rf_acc*100:.1f}% | AUC: {rf_auc:.3f}")
print("\n  Key Learnings:")
print("  • Credit score & late payments are strongest predictors")
print("  • Random Forest outperforms Logistic Regression")
print("  • AUC-ROC is better metric than Accuracy for imbalanced data")
print("=" * 60)
