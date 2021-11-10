try:
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union

    import pandas as pd
    from PIL import Image
    import streamlit as st
    import os
except Exception as e:
    print(e)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
enabled = False

def upload_image():
    uploaded_image = st.file_uploader("Upload image:", type=["png", "jpg", 'jpeg'])
    if uploaded_image is not None:
        st.write("Original image: " + uploaded_image.name)
        image_file = load_image(uploaded_image)
        st.image(image_file, width=700)
        with open(os.path.join(BASE_DIR, "data/uploads", uploaded_image.name),"wb") as f:
            f.write(uploaded_image.getbuffer())
            enabled = True
    return uploaded_image

def sider_bar():
    mod_list = ["ResNet", "VGG", "AlexNet"]
    container = st.sidebar.container()
    all = st.sidebar.checkbox("All Models", value=False)
    if all:
        selected_options = container.multiselect("Models options:", mod_list, mod_list)
    else:
        selected_options = container.multiselect("Models options:", mod_list)
    return container

@st.cache
def load_image(image_file):
    image = Image.open(image_file)
    return image

def show_prediction(image_file, model):
    col1, col2 = st.columns(2)
    original = Image.open(image_file)
    col1.header("Original:")
    col1.image(original, use_column_width=True)

    grayscale = original.convert('LA')
    col2.header("Predicted:")
    col2.image(grayscale, use_column_width=True)
    st.write("Confidence Score: ")


def main():
    uploaded_image = upload_image()
    container = sider_bar()

    if st.sidebar.button('Predict'):
        show_prediction(uploaded_image, model="VGG")

if __name__ == "__main__":
    main()