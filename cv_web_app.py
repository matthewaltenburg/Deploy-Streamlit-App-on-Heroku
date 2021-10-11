# Filename: cv_web_app.py

import streamlit as st
import numpy as np
import onnxruntime
import torch
from PIL import Image
from torchvision import transforms


def transform_image(image):
    """ This function transforms an image
    in order to crop it to pre-defined size
    and then normalized. """

    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    return transform(image)


def model_inference(model, img_pil):
    """ This function carries out model
    inference using the ONNX model path
    and image as input."""
    # Transform image
    img_t = transform_image(img_pil)
    batch_t = torch.unsqueeze(img_t, 0)

    # Create ONNX runtime session for model
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # Carry out model inference
    result = session.run([output_name], {input_name: batch_t.numpy()})[0]

    # Get class names
    with open('/home/matthew/Downloads/Deploy-Streamlit-App-on-Heroku/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # Get class name
    class_name = classes[np.argmax(result.ravel())]

    return class_name


st.title("Image Classification")

# Disable warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# ONNX model file path
model = "model.onnx"

# Create a button for loading an image
image_path = st.file_uploader("", type=["png", "jpg", "jpeg"])

if image_path is not None:
    # Read image
    image = Image.open(image_path)

    # Carry inference
    class_name = model_inference(model, image)

    # Display image
    st.image(image, use_column_width=True)

    st.title("Class recognized: {}".format(class_name))
