import os
os.environ["PORT"] = "8501"

import streamlit as st
import numpy as np
from PIL import Image
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import tempfile
import gdown

# ------------------------ Model Setup ------------------------

MODEL_URL = "https://drive.google.com/uc?id=1J9sne4__yo5vA9ZqOY2KQeXrPgcHZf0H"
MODEL_PATH = "RealESRGAN_x4plus.pth"

# Auto-download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("📦 Downloading ESRGAN model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = RRDBNet(num_in_ch=3, num_out_ch=3, nf=64, nb=23, gc=32,
                upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1,
                upsample_mode='upconv')

upscaler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_available() else False
)

# ------------------------ Streamlit UI ------------------------

st.set_page_config(page_title="📸 Photo Enhancer", layout="centered")
st.title("✨ AI 4K Photo Enhancer")
st.caption("Upscale blurry or low-resolution images using Real-ESRGAN.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Original", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name

    with st.spinner("🔧 Enhancing image..."):
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)
        try:
            output, _ = upscaler.enhance(img_np)
            st.image(output, caption="Enhanced", use_column_width=True)
        except Exception as e:
            st.error(f"Enhancement failed: {e}")
