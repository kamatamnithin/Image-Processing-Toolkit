import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -------------------
# Helper Functions
# -------------------
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def get_image_info(img, image_file):
    h, w, c = img.shape if len(img.shape) == 3 else (*img.shape, 1)
    file_size = len(image_file.getvalue()) / 1024  # KB
    return {
        "Dimensions": f"{w} x {h}",
        "Channels": c,
        "File Size (KB)": f"{file_size:.2f}",
        "Format": image_file.type
    }

def convert_color(img, mode):
    if mode == "Grayscale":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif mode == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif mode == "YCbCr":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif mode == "BGR":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def apply_transformation(img, transform, angle=0, scale=1.0, tx=0, ty=0):
    h, w = img.shape[:2]

    if transform == "Rotate":
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    elif transform == "Scale":
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    elif transform == "Translate":
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h))

    elif transform == "Affine":
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(img, M, (w, h))

    elif transform == "Perspective":
        pts1 = np.float32([[50, 50], [w-50, 50], [50, h-50], [w-50, h-50]])
        pts2 = np.float32([[10, 100], [w-100, 50], [100, h-100], [w-50, h-50]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h))

    return img

def apply_filter(img, filter_type, ksize=5):
    if filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif filter_type == "Median Blur":
        return cv2.medianBlur(img, ksize)
    elif filter_type == "Mean Blur":
        return cv2.blur(img, (ksize, ksize))
    elif filter_type == "Sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        return cv2.convertScaleAbs(cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0))
    elif filter_type == "Laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)
    return img

def apply_morphology(img, morph_type, ksize=5):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((ksize, ksize), np.uint8)

    if morph_type == "Dilation":
        return cv2.dilate(gray, kernel, iterations=1)
    elif morph_type == "Erosion":
        return cv2.erode(gray, kernel, iterations=1)
    elif morph_type == "Opening":
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif morph_type == "Closing":
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return gray

# -------------------
# Streamlit GUI
# -------------------
st.set_page_config(page_title="Image Processing Toolkit", layout="wide")
st.title("🖼 Image Processing & Analysis Toolkit")

image_file = st.sidebar.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if image_file is not None:
    img = load_image(image_file)

    st.sidebar.header("⚙️ Operations")
    operation = st.sidebar.selectbox(
        "Choose an operation",
        ["Show Image Info", "Color Conversion", "Transformations", "Filtering & Morphology"]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    processed_img = img.copy()

    if operation == "Show Image Info":
        info = get_image_info(img, image_file)
        st.sidebar.subheader("📊 Image Info")
        for k, v in info.items():
            st.sidebar.write(f"**{k}**: {v}")

    elif operation == "Color Conversion":
        mode = st.sidebar.radio("Select Conversion", ["Grayscale", "HSV", "YCbCr", "BGR"])
        processed_img = convert_color(img, mode)

    elif operation == "Transformations":
        transform = st.sidebar.radio("Select Transformation", ["Rotate", "Scale", "Translate", "Affine", "Perspective"])

        if transform == "Rotate":
            angle = st.sidebar.slider("Rotation Angle", -180, 180, 45)
            processed_img = apply_transformation(img, "Rotate", angle=angle)

        elif transform == "Scale":
            scale = st.sidebar.slider("Scaling Factor", 0.1, 2.0, 1.0)
            processed_img = apply_transformation(img, "Scale", scale=scale)

        elif transform == "Translate":
            tx = st.sidebar.slider("Shift X", -100, 100, 20)
            ty = st.sidebar.slider("Shift Y", -100, 100, 20)
            processed_img = apply_transformation(img, "Translate", tx=tx, ty=ty)

        elif transform == "Affine":
            processed_img = apply_transformation(img, "Affine")

        elif transform == "Perspective":
            processed_img = apply_transformation(img, "Perspective")

    elif operation == "Filtering & Morphology":
        choice = st.sidebar.radio("Select Filter/Morphology", 
                                  ["Gaussian Blur", "Median Blur", "Mean Blur", "Sobel", "Laplacian", 
                                   "Dilation", "Erosion", "Opening", "Closing"])
        ksize = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)

        if choice in ["Gaussian Blur", "Median Blur", "Mean Blur", "Sobel", "Laplacian"]:
            processed_img = apply_filter(img, choice, ksize)
        else:
            processed_img = apply_morphology(img, choice, ksize)

    with col2:
        st.subheader("Processed Image")
        if processed_img.ndim == 2:
            st.image(processed_img, use_container_width=True, channels="GRAY")
        else:
            st.image(processed_img, use_container_width=True)

    if st.sidebar.button("💾 Save Processed Image"):
        save_path = "processed_image.png"
        cv2.imwrite(save_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR) 
                    if processed_img.ndim == 3 else processed_img)
        st.sidebar.success(f"Image saved as {save_path}")
else:
    st.info("👆 Upload an image to get started")
    
