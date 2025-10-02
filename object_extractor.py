import warnings
warnings.filterwarnings("ignore")  # Hide Python warnings

import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # Suppress Torch/CUDA info logs

import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2

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
    "Choose an image (JPG, JPEG, PNG, etc.)",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
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

# ------------------ Model Setup ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(pretrained=False, num_classes=2)
LONG_SIDE_SIZE = 640 

# Update MODEL_PATH to your trained model
MODEL_PATH = "C:/Users/laksh/Downloads/best_deeplabv3_finetuned.pth"
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

# ------------------ Helper Functions ------------------
resize_input_size = (256, 256)  # input for model
display_max_width = 1800        # increase width for original & masked images

def preprocess_image(img):
    transform = T.Compose([
        T.Resize(resize_input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor.to(DEVICE), img

def clean_mask(mask):
    """Cleans the raw mask with morphology + Gaussian blur for smooth edges"""
    mask_uint8 = (mask*255).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    clean = cv2.GaussianBlur(clean, (7,7), 0)
    return clean

def apply_mask_to_image(original_img, mask):
    """Applies mask to original image with black background"""
    mask_resized = cv2.resize(mask, (original_img.width, original_img.height), interpolation=cv2.INTER_LINEAR)
    img_array = np.array(original_img)
    out = np.zeros_like(img_array)  # black background
    out[mask_resized>127] = img_array[mask_resized>127]  # keep subject
    return Image.fromarray(out)

def resize_for_display(img, max_width=display_max_width):
    """Resize image for display while maintaining aspect ratio"""
    w_percent = (max_width / float(img.width))
    h_size = int(float(img.height) * w_percent)
    return img.resize((max_width, h_size))

# ------------------ Main Processing ------------------
if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    img_tensor, _ = preprocess_image(original_image)

    # Predict mask
    with torch.no_grad():
        output = model(img_tensor)['out']
        mask_raw = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    mask_cleaned = clean_mask(mask_raw)
    masked_image = apply_mask_to_image(original_image, mask_cleaned)

    # Resize for display
    original_display = resize_for_display(original_image)
    masked_display = resize_for_display(masked_image)

    # Display Original | Masked
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼 Original Image:")
        st.image(original_display, use_container_width=False)  # <-- force full pixel size
    with col2:
        st.subheader("✨ Masked Image:")
        st.image(masked_display, use_container_width=False)    # <-- force full pixel size
        buf = io.BytesIO()
        masked_image.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Masked Image",
            data=buf.getvalue(),
            file_name=f"masked_{uploaded_file.name}",
            mime="image/png"
        )

    # ------------------ Overlay Section (Bigger) ------------------
    st.markdown("---")
    st.subheader("🔄 Overlay Image:")
    opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5)
    overlay = Image.blend(original_image, masked_image, alpha=opacity)

    # ---- Resize overlay to bigger size ----
    overlay_width = 1200  # bigger overlay
    w_percent = overlay_width / float(overlay.width)
    h_size = int(float(overlay.height) * w_percent)
    overlay_display = overlay.resize((overlay_width, h_size))

    # ---- Center-align overlay ----
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.image(overlay_display, caption=f'Overlay ({opacity*100:.0f}%)', use_container_width=False)

    # ---- Download overlay ----
    buf_overlay = io.BytesIO()
    overlay.save(buf_overlay, format="PNG")
    st.download_button(
        label="📥 Download Overlay Image",
        data=buf_overlay.getvalue(),
        file_name=f"overlay_{uploaded_file.name}",
        mime="image/png"
    )

else:
    st.info("Please upload an image to start processing.")
