import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

X = np.load("data/X_mfcc.npy").reshape(-1, 40, 1)
y = np.load("data/y_labels.npy")
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=(40, 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')  # auto-adjust to correct class count
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30, batch_size=32, validation_split=0.2)

model.save("models/cnn_model.h5")
