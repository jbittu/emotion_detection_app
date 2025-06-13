import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
# Load features
X = np.load("data/X_mfcc.npy")  # shape: (samples, features)
if X.ndim == 3:
    X = X.reshape(X.shape[0], -1)  # Flatten for scikit-learn

# Load labels
y = np.load("data/y_labels.npy")
print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)

# Check if labels are valid
if y.size == 0:
    raise ValueError("Label array y is empty. Check data/y_labels.npy.")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Save label encoder
with open("models/rf_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/rf_model.pkl")
