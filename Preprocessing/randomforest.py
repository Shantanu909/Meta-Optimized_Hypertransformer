import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('merged.csv')

# Compute central frequency and bandwidth
data['central_frequency'] = (data['fstart'] + data['fend']) / 2
data['bandwidth'] = data['fend'] - data['fstart']

# Select features
selected_columns = ['frequency', 'central_frequency', 'bandwidth', 'amplitude', 'snr', 'phase', 'label']
data_selected = data[selected_columns]

# Split features and labels
X = data_selected.iloc[:, :-1]
y = data_selected['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)
print(X_train.shape,X_test.shape)

from tqdm import tqdm  # Keep this import at the top

# Initialize model with warm_start so we can increment trees manually
rf_model = RandomForestClassifier(
    n_estimators=1,
    max_depth=6,
    warm_start=True,
    random_state=42,
    class_weight='balanced'
)

# Manual training loop with progress bar
n_total_trees = 1000
for i in tqdm(range(1, n_total_trees + 1), desc="Training Trees"):
    rf_model.n_estimators = i
    rf_model.fit(X_train, y_train)



y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Feature importance plot
importances = rf_model.feature_importances_
indices = importances.argsort()

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(X.shape[1]), importances[indices], align="center")
plt.yticks(range(X.shape[1]), X.columns[indices])
plt.xlabel("Relative Importance")
plt.show()

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
