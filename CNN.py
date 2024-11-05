
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv("C:\\Users\\safin\\OneDrive\\Desktop\\ML2\\EngineFaultDB_Final.csv")

# Separate features and target variable
X = data.drop('Fault', axis=1).values
y = data['Fault'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the input data to be a 3D array for Conv1D (samples, timesteps, features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model configuration
input_shape = (X_train.shape[1], 1)  # Shape for Conv1D (timesteps, features)
num_classes = 4
# Designing the 1D CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Subplot for Model Accuracy and Loss
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Accuracy subplot
axs[0].plot(history.history['accuracy'], label='Train Accuracy')
axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(loc='upper left')

# Loss subplot
axs[1].plot(history.history['loss'], label='Train Loss')
axs[1].plot(history.history['val_loss'], label='Validation Loss')
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(loc='upper left')

plt.tight_layout()
plt.show()

# Continue with the rest of your code for predictions, confusion matrix, and classification report

# Predict the values from the test dataset
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)  # Convert predictions to class indices

# Confusion Matrix Calculation
cm = confusion_matrix(y_test, y_pred_classes)

# Visualization of Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fault', 'R M', 'L M', 'L V'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_test, y_pred_classes, target_names=['No Fault', 'Rich Mixture', 'Lean Mixture', 'Low Voltage'])
print(report)

