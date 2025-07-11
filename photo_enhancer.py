import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import tempfile
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="🖼️ AI Photo Editor", layout="centered")
st.title("🎨 All-in-One AI Photo Editor")
st.caption("Upload and enhance your photos using AI tools")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

with st.sidebar:
    st.header("⚙️ Editing Options")
    enhance_ai = st.checkbox("🧠 Enhance to 4K (Real-ESRGAN)")
    color = st.slider("🌈 Color", -20.0, 20.0, 1.0)
    sharpness = st.slider("✏️ Sharpness", -20.0, 20.0, 1.0)
    brightness = st.slider("🌟 Brightness", -20.0, 20.0, 1.0)
    contrast = st.slider("🌗 Contrast", -20.0, 20.0, 1.0)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    # Use exif_transpose to display the image in correct orientation
    img = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
    st.image(img, caption="🖼️ Original Image", use_container_width=True)

    # Apply Enhancements
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # Optional Real-ESRGAN 4K Enhancement
    if enhance_ai:
        with st.spinner("Upscaling with Real-ESRGAN..."):
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            upsampler = RealESRGANer(
                scale=4,
                model_path="RealESRGAN_x4plus.pth",
                model=model,
                tile=128,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
                device=device
            )
            img_np = np.array(img)
            output, _ = upsampler.enhance(img_np, outscale=4)
            img = Image.fromarray(output)

    st.image(img, caption="✅ Final Output", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_out:
        img.save(tmp_out.name)
        st.download_button("⬇️ Download Edited Image", open(tmp_out.name, "rb").read(), "edited_photo.png")
