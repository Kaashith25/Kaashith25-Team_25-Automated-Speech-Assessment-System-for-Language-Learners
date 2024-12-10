import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# Load preprocessed data
X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")
X_test = pd.read_csv("X_test.csv")

y_train_reg = pd.read_csv("y_train_reg.csv")
y_val_reg = pd.read_csv("y_val_reg.csv")
y_test_reg = pd.read_csv("y_test_reg.csv")

y_train_class = pd.read_csv("y_train_class.csv")
y_val_class = pd.read_csv("y_val_class.csv")
y_test_class = pd.read_csv("y_test_class.csv")

# Define the model
def build_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # For regression output (fluency score)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and train the regression model (Fluency Score prediction)
model_reg = build_model(X_train.shape[1])

# Train the regression model
model_reg.fit(X_train, y_train_reg, epochs=50, validation_data=(X_val, y_val_reg), batch_size=32)

# Save the trained model
model_reg.save("fluency_score_model.h5")
print("Regression model trained and saved!")

# For proficiency classification, define another model with a softmax output
def build_classification_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(5, activation='softmax')  # 5 CEFR levels (A1, A2, B1, B2, C1)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the classification model (Proficiency Level prediction)
model_class = build_classification_model(X_train.shape[1])

# Train the classification model
model_class.fit(X_train, y_train_class, epochs=50, validation_data=(X_val, y_val_class), batch_size=32)

# Save the trained model
model_class.save("proficiency_level_model.h5")
print("Classification model trained and saved!")

# Evaluate Regression Model (Fluency)
test_loss_reg = model_reg.evaluate(X_test, y_test_reg)
print(f"Test Loss (Fluency Score): {test_loss_reg}")

# Evaluate Classification Model (Proficiency Level)
test_loss_class, test_acc_class = model_class.evaluate(X_test, y_test_class)
print(f"Test Loss (Proficiency Level): {test_loss_class}")
print(f"Test Accuracy (Proficiency Level): {test_acc_class}")

# Predict Fluency Scores (Regression)
predictions_reg = model_reg.predict(X_test)

# Predict Proficiency Levels (Classification)
predictions_class = model_class.predict(X_test)
predicted_classes = predictions_class.argmax(axis=-1)  # Get the class with highest probability
