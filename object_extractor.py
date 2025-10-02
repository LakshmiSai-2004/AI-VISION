import streamlit as st
from PIL import Image
import numpy as np
import io
import zipfile
from rembg import remove  # Using REMBG instead of Remove.bg

# ------------------ Resize Function ------------------
def resize_image(img, max_width=600):
    w_percent = (max_width / float(img.width))
    h_size = int((float(img.height) * float(w_percent)))
    return img.resize((max_width, h_size))

# ------------------ Streamlit Page Config ------------------
st.set_page_config(
    page_title="🎨 AI Object Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Sidebar ------------------
st.sidebar.header("Customize Dashboard")
bg_color = st.sidebar.color_picker("Dashboard Background", "#E6E6FA")
sidebar_start = st.sidebar.color_picker("Sidebar Gradient Start", "#FFB6C1")
sidebar_end = st.sidebar.color_picker("Sidebar Gradient End", "#87CEFA")

st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

st.sidebar.markdown("""
**Instructions:**
- Upload one image at a time.
- See original and masked images side by side.
- Use overlay slider to blend images.
- Download masked image.
""")

# ------------------ Custom CSS ------------------
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg_color};
}}
h1 {{
    text-align: center;
    color: white;
    font-family: 'Arial', sans-serif;
}}
.css-1d391kg {{
    background: linear-gradient(to bottom, {sidebar_start}, {sidebar_end});
    color: white;
    font-weight: bold;
}}
h2, h3 {{
    color: #4B0082;
}}
.stButton button {{
    background-color: #FF69B4;
    color: white;
    border-radius: 10px;
    height: 40px;
    width: 220px;
    font-size: 16px;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown(f"""
<div style='text-align: center; padding: 20px; 
            background: linear-gradient(to right, {sidebar_start}, {sidebar_end});
            color: white; border-radius:10px;' >
    <h1>🤖 AI Object Extractor Dashboard</h1>
    <p style='text-align: center; font-size:18px; color:white;' >
    Upload images, extract objects, and download results!</p>
    <hr style='border:2px solid white'>
</div>
""", unsafe_allow_html=True)

# ------------------ Main Processing ------------------
if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")

    # Using REMBG to remove background
    masked_image = remove(original_image)
    
    # Convert background to black
    data = np.array(masked_image)
    alpha_channel = data[..., 3]
    data[..., :3][alpha_channel == 0] = [0, 0, 0]  # Black background
    data[..., 3] = 255
    masked_image = Image.fromarray(data)

    original_resized = resize_image(original_image, max_width=800)
    masked_resized = resize_image(masked_image, max_width=800)

    # Row 1: Original | Masked
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🖼 Original: {uploaded_file.name}")
        st.image(original_resized, width=600)
    with col2:
        st.subheader(f"✨ Masked: {uploaded_file.name}")
        st.image(masked_resized, width=600)

        buf = io.BytesIO()
        masked_image.save(buf, format="PNG")
        st.download_button(
            label=f"📥 Download Masked",
            data=buf.getvalue(),
            file_name=f"masked_{uploaded_file.name}",
            mime="image/png"
        )

    st.markdown("---")

    # Row 2: Overlay Slider
    st.subheader(f"🔄 Overlay for: {uploaded_file.name}")
    opacity = st.slider(f"Overlay Opacity", 0.0, 1.0, 0.5)
    overlay = Image.blend(original_resized, masked_resized.convert("RGB"), alpha=opacity)
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.image(overlay, width=600, caption=f"Overlay ({opacity*100:.0f}%)")

else:
    st.info("Please upload an image to start processing.")
