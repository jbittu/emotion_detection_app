import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

X = np.load("data/X_mfcc.npy")
y = np.load("data/y_labels.npy")
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2)
model = SVC(probability=True)
model.fit(X_train, y_train)

joblib.dump(model, "models/svm_model.pkl")
