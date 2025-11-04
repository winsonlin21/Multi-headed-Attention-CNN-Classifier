import os
import argparse
from typing import Optional
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from training import BaseCNN, AttentionCNN


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(in_channels: int = 1, num_features: int = 256, num_classes: int = 10, num_heads: int = 4, device=None):
    base = BaseCNN(in_channels=in_channels)
    model = AttentionCNN(base, num_features=num_features, num_classes=num_classes, num_heads=num_heads)
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str] = None, device=None):
    if device is None:
        device = get_device()

    if checkpoint_path is None:
        # prefer best, fallback to last
        best = os.path.join("./checkpoints", "att_best.pt")
        last = os.path.join("./checkpoints", "att_last.pt")
        checkpoint_path = best if os.path.exists(best) else (last if os.path.exists(last) else None)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No checkpoint found. Expected att_best.pt or att_last.pt in ./checkpoints/")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with Attention-based CNN model.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint.')
    parser.add_argument('--image', type=str, required=False, default=None, help='Path to input image. If omitted the script will look for ./image.jpg')
    args = parser.parse_args()

    device = get_device()
    model = build_model(device=device)
    model = load_checkpoint(model, args.checkpoint, device=device)

    image_path = args.image
    if image_path is None:
        default_candidate = os.path.join(os.getcwd(), 'image.jpg')
        prompt = (
            "Enter path to image file (or press Enter to use './image.jpg' if present; type 'q' to quit): "
        )
        while True:
            user_input = input(prompt).strip()
            # user wants default
            if user_input.lower() in ("q", "quit", "exit"):
                print("Aborted by user.")
                raise SystemExit(1)

            candidate = user_input.strip('"')
            if os.path.exists(candidate):
                image_path = candidate
                # preprocessing image
                input_size = (28, 28)  # change as needed
                preprocess = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: 1.0 - t),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                image = Image.open(image_path)
                input_tensor = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_class = output.argmax(dim=1).item()

                print(f"Predicted class: {predicted_class}")
            else:
                print(f"Path not found: {candidate}\nPlease try again or type 'q' to quit.")

