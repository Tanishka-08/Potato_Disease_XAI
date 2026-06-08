import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load trained model
model = load_model('models/efficientnet_model.h5')

# Class labels
class_names = ['early_blight', 'healthy', 'late_blight']

# Load image
img_path = img_path = img_path = img_path = r"C:\Users\Admin\OneDrive\Desktop\Potato_Disease_XAI\dataset\test\early_blight\daa35c03-7307-4e2e-8b15-f3c5d8f.png.JPG"

img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img)

# Store original image
original_img = img_array.astype('uint8')

# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# Preprocess image
img_array = preprocess_input(img_array)

# Prediction
prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)

print("Predicted Class:", class_names[predicted_class])

# LIME Explainer
explainer = lime_image.LimeImageExplainer()

# Prediction function for LIME
def predict_fn(images):
    images = preprocess_input(images.copy())
    return model.predict(images)

# Generate explanation
explanation = explainer.explain_instance(
    original_img,
    predict_fn,
    top_labels=3,
    hide_color=0,
    num_samples=1000
)

# Get explained image
temp, mask = explanation.get_image_and_mask(
    predicted_class,
    positive_only=True,
    num_features=5,
    hide_rest=False
)

# Plot result
plt.figure(figsize=(8,8))

plt.imshow(mark_boundaries(temp / 255.0, mask))

plt.title(f"LIME Explanation - {class_names[predicted_class]}")

# Save output
plt.savefig("outputs/lime/lime_explanation.png")

plt.show()