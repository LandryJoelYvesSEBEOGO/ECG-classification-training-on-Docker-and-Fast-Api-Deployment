import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from joblib import dump

df= pd.read_csv(r"ecg.csv")

# Generate column names F1, F2, ..., Fn
n_columns = df.shape[1]  # Total number of columns
column_names = [f"F{i}" for i in range(1, n_columns)] + ["label"]

# Assign the column names to the DataFrame
df.columns = column_names


# Supposons que df est votre DataFrame et 'label' est la colonne cible
X = df.drop(columns=['label'])  # Features
y = df['label']  # Target

# Diviser les données en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (Standardization: zero mean, unit variance)
#scaler = StandardScaler()

# Fit scaler only on the training set, then transform both sets
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Ensure the target labels are in the correct shape
y_train = y_train.values.reshape(-1, 1)  # Ensure shape is (n_samples, 1)
y_test = y_test.values.reshape(-1, 1)    # Ensure shape is (n_samples, 1)


# Define the number of classes in your problem
num_classes = len(y.unique())  # Replace 'y' with your target data if necessary

# Determine the activation function for the output layer
if num_classes == 2:  # Binary classification
    output_activation = 'sigmoid'
    loss_function = 'binary_crossentropy'
    
else:  # Multiclass classification
    output_activation = 'softmax'
    loss_function = 'sparse_categorical_crossentropy'

# Create a two-layer neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
    Dense(64, activation='relu'),  # Second hidden layer
    Dense(1, activation=output_activation)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,  # Training data
    validation_split=0.2,    # Use 20% of training data for validation
    epochs=50,               # Number of training epochs
    batch_size=4,           # Batch size
    verbose=1                # Show progress during training
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predictions to class labels
if num_classes == 2:  # Binary classification
    y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
else:  # Multiclass classification
    y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Save the model to a file using joblib
dump(model, 'model.pkl')

    