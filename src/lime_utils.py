import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.efficientnet import preprocess_input

from lime import lime_image
from skimage.segmentation import mark_boundaries


def generate_lime(model, img_array, predicted_class):
    """
    Generates a LIME explanation for an image.

    Parameters:
        model : Trained Keras model
        img_array : Image array (224x224x3)
        predicted_class : Predicted class index

    Returns:
        Matplotlib Figure
    """

    # Convert image to uint8
    original_image = img_array.astype(np.uint8)

    # Prediction function for LIME
    def predict_fn(images):
        images = preprocess_input(images.copy())
        return model.predict(images, verbose=0)

    # Create explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation
    explanation = explainer.explain_instance(
        original_image,
        predict_fn,
        top_labels=3,
        hide_color=0,
        num_samples=1000
    )

    # Generate mask
    temp, mask = explanation.get_image_and_mask(
        predicted_class,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.axis("off")
    ax.set_title("LIME Explanation")

    return fig