import pandas as pd
from PIL import Image
import streamlit as st
import os

from streamlit_drawable_canvas import st_canvas

@st.cache
def load_image(image_file):
    image = Image.open(image_file)
    return image



canvas = st.canvas(size=(200, 200))
image_url=r'C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\images\9001400L.png'
bg = load_image(image_url)
canvas.draw_image(bg, 50, 50)

canvas.fill_rect(0, 0, 50, 50)

def handle_mouse_down(x, y):
    # Do something else
    pass
canvas.on_mouse_down(handle_mouse_down)