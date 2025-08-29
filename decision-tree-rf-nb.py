import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score   # ✅ added accuracy_score

# ----------------------------
# 1. Load dataset from Excel
# ----------------------------
excel_file = "decision-tree-dataset.xlsx"  # replace with your file path
df = pd.read_excel(excel_file)

# ----------------------------
# 2. Encode categorical features and labels
# ----------------------------
X = df.drop(columns=["Part", "Label"])
y = df["Label"]

# Encode all features
for col in X.columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Encode labels
le_label = LabelEncoder()
y_encoded = le_label.fit_transform(y)
label_names = le_label.classes_

# ----------------------------
# 3. Train/Test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded   # ✅ ensure all groups are represented
)

# ----------------------------
# 4. Define classifiers
# ----------------------------
classifiers = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
    "Categorical Naive Bayes": CategoricalNB()
}

# ----------------------------
# 5. Train classifiers and collect results
# ----------------------------
results = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ✅ Compute accuracy for the whole model
    acc = accuracy_score(y_test, y_pred)

    # Classification report (per-class metrics)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(label_names))),  # include all classes
        target_names=label_names,
        output_dict=True
    )
    
    # Add each group’s precision/recall/f1/support + overall accuracy
    for label in label_names:
        results.append({
            "Model": name,
            "Group": label,
            "Precision": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1-score": report[label]["f1-score"],
            "Support": report[label]["support"],
            "Accuracy": acc   # ✅ add accuracy column
        })

# ----------------------------
# 6. Create summary DataFrame
# ----------------------------
summary_df = pd.DataFrame(results)
print(summary_df)
