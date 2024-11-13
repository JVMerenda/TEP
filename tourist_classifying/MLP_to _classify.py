import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

# --- Step 1: Load and prepare the train data ---
directory_train='/home/DATA/datasets/SIS_teps/'
directory_class='/home/DATA/datasets/SIS_teps/'
directory_test='/home/DATA/datasets/SIS_teps/'
directory_output='/home/DATA/datasets/SIS_teps/'
train_data = pd.read_csv(directory_train+'train_data.csv').values
train_labels = pd.read_csv(directory_class+'classes.csv').values


# One-hot encoding of classes (20 classes)
encoder = OneHotEncoder(sparse=False)
train_labels = encoder.fit_transform(train_labels)

# Data normalization
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

# --- Step 2: train/validation split (75% trein, 25% validation) ---
#We'll use validation process to set an apropriate learning rate
X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_labels, test_size=0.25, random_state=42
)

# --- Model creation ---
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(20, activation='softmax')  # 20 output classes
    ])
    return model

# --- Step 3: Validation to find the best learning rate ---
learning_rates = [1e-4, 1e-3, 0.5e-2, 1e-2, 0.5e-1, 1e-1]
best_accuracy = 0
best_lr = None

for lr in learning_rates:
    model = create_model()
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                        validation_data=(X_val, y_val), verbose=0)
    
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"Learning rate: {lr}, Validation Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_lr = lr

print(f"Best Learning Rate: {best_lr}")

# --- Step 4: training over all training data using the best learning rate ---
model = create_model()
optimizer = Adam(learning_rate=best_lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# --- Step 5: Load and prepare the test dataset
test_data = pd.read_csv('test_data.csv').values
test_names = test_data[:, 0]  # first column are the data name (as they are in \SIS_teps)
test_features = scaler.transform(test_data[:, 1:].astype(float))  # Others columns are features


# --- Step 6: make predictions about the test data ---
predictions = model.predict(test_features)
predicted_classes = encoder.inverse_transform(predictions)

# --- Step 7: save the predictions into a .txt file ---
with open(directory_output+'predicoes.txt', 'w') as f:
    for name, pred in zip(test_names, predicted_classes):
        f.write(f"{name}: {pred[0]}\n")

