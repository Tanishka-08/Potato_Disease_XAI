import os
import numpy as np
import shap
import matplotlib.pyplot as plt

from tensorflow.keras.applications.efficientnet import preprocess_input


def generate_shap(model, img_array):

    # Create output folder if it doesn't exist
    os.makedirs("outputs/shap", exist_ok=True)

    # Original image
    original_img = img_array.astype(np.uint8)

    # Model input
    input_img = np.expand_dims(original_img, axis=0)
    input_img = preprocess_input(input_img.copy())

    # SHAP Masker
    masker = shap.maskers.Image(
        "inpaint_telea",
        original_img.shape
    )

    # SHAP Explainer
    explainer = shap.Explainer(
        model,
        masker
    )

    # Generate SHAP values
    shap_values = explainer(
        input_img,
        max_evals=500,
        batch_size=50
    )

    # Predicted class
    predicted_class = np.argmax(
        model.predict(input_img, verbose=0)
    )

    # Plot SHAP
    shap.image_plot(
        [shap_values.values[..., predicted_class]],
        pixel_values=original_img[np.newaxis, ...],
        show=False
    )

    # Save image
    plt.savefig(
        "outputs/shap/shap_explanation.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    return "outputs/shap/shap_explanation.png"