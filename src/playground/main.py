import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from PIL import Image
import open_clip


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    return model, preprocess


model, preprocess = load_model()
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def predict():
    img_path = "docs/CLIP.png"
    img = Image.open(img_path)
    image = preprocess(img).unsqueeze(0)

    labels = ["a diagram", "a dog", "a cat"]
    text = tokenizer(labels)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Labels: ", labels)
    print("Label probs:", np.round(text_probs, 2))  # prints: [[1., 0., 0.]]


predict()
