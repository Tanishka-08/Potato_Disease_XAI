from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import test_generator

# Load trained model
model = load_model('models/efficientnet_model.h5')

# Predict on test data
predictions = model.predict(test_generator)

# Convert probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = test_generator.classes

# Class names
class_labels = list(test_generator.class_indices.keys())

# Classification Report
print("\nClassification Report:\n")

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
)

print(report)

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save confusion matrix
plt.savefig("outputs/graphs/confusion_matrix.png")

plt.show()