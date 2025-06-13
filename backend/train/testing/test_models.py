import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import tensorflow as tf

# Load features and labels
X = np.load("data/X_mfcc.npy")
y = np.load("data/y_labels.npy")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- SVM ---
svm_model = joblib.load("models/svm_model.pkl")
svm_pred = svm_model.predict(X_test)
print(" SVM Results:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred, target_names=le.classes_))

# --- Random Forest ---
rf_model = joblib.load("models/rf_model.pkl")
rf_pred = rf_model.predict(X_test)
print("\n Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# --- CNN ---
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
X_test_cnn = X_test.reshape(-1, 40, 1)
cnn_pred = cnn_model.predict(X_test_cnn)
cnn_pred_labels = np.argmax(cnn_pred, axis=1)
print("\n CNN Results:")
print("Accuracy:", accuracy_score(y_test, cnn_pred_labels))
print(confusion_matrix(y_test, cnn_pred_labels))
print(classification_report(y_test, cnn_pred_labels, target_names=le.classes_))

# --- LSTM ---
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
lstm_pred = lstm_model.predict(X_test_cnn)
lstm_pred_labels = np.argmax(lstm_pred, axis=1)
print("\n LSTM Results:")
print("Accuracy:", accuracy_score(y_test, lstm_pred_labels))
print(confusion_matrix(y_test, lstm_pred_labels))
print(classification_report(y_test, lstm_pred_labels, target_names=le.classes_))

# --- GRU ---
try:
    X_seq = np.load("data/X_mfcc.npy")
except FileNotFoundError:
    print("\n GRU not tested: 'data/X_mfcc.npy' not found.")
else:
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_encoded, test_size=0.2, random_state=42)

    gru_model = tf.keras.models.load_model("models/gru_model.keras")
    gru_pred = gru_model.predict(X_seq_test)
    gru_pred_labels = np.argmax(gru_pred, axis=1)

    print("\n GRU Results:")
    print("Accuracy:", accuracy_score(y_seq_test, gru_pred_labels))
    print(confusion_matrix(y_seq_test, gru_pred_labels))
    print(classification_report(y_seq_test, gru_pred_labels, target_names=le.classes_))
