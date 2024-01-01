import gradio
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

from core.config import load_config


def binary_predictions(raw_preds: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(raw_preds) >= 0.5).float()


def blur_background(input_image):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    # Generate a blank mask
    image_tensor = preprocess(input_image).unsqueeze(0)
    mask = binary_predictions(model(image_tensor)).numpy()[0].transpose(1, 2, 0)

    input_image = cv2.resize(input_image, mask.shape[:2])

    # apply a strong Gaussian blur to the areas outside the mask
    blurred = cv2.GaussianBlur(input_image, (51, 51), 0)
    result = np.where(mask, input_image, blurred)

    # Convert the result back to RGB format for Gradio
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


if __name__ == "__main__":
    config = load_config("demo")
    model = torch.jit.load(config["model_path"])

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((240, 240), antialias=True),
        ]
    )

    ui = gradio.Interface(
        fn=blur_background,
        inputs=gradio.Image(sources=["webcam"], streaming=True),
        outputs="image",
        live=True,
        title="Image segmentation demo!",
        allow_flagging="never",
    )
    ui.dependencies[0]["show_progress"] = False
    ui.launch()
