import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
X = np.load("data/X_mfcc.npy").reshape(-1, 40, 1)
y = np.load("data/y_labels.npy")

# Check label validity
if y.size == 0:
    raise ValueError("Label array y is empty. Please regenerate data/y_labels.npy.")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

# Build GRU model
model = Sequential([
    GRU(64, input_shape=(40, 1), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30, batch_size=32, validation_split=0.2)

model.save("models/gru_model.h5")
