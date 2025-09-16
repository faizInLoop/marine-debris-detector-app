import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
from collections import Counter
import base64
import pandas as pd
import os
from datetime import datetime
import io
import matplotlib.pyplot as plt
import requests   # 👈 added for downloading model

# STYLING
def set_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        st.markdown(f"<style>.stApp {{background-image: url('data:image/jpeg;base64,{encoded_string}'); background-attachment: fixed; background-size: cover;}}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Background image '{image_file}' not found.")

st.set_page_config(page_title="Marine Debris Detector", page_icon="🌊", layout="centered")
set_bg_from_local("bg.jpg")

# MODEL LOADING
MODEL_URL = "https://huggingface.co/FaizAI/marine-debris-detector/resolve/main/best.pt"
MODEL_PATH = "best.pt"

# Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner("Downloading model from Hugging Face..."):
                r = requests.get(MODEL_URL, stream=True)
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success("Model downloaded successfully ✅")
        except Exception as e:
            st.error(f"Failed to download model: {e}")

download_model()

@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'best.pt' is available.")
        return None

model = load_model(MODEL_PATH)

# SIDEBAR AND DASHBOARD
st.sidebar.title("📊 Analytics Dashboard")
HISTORY_FILE = 'detection_history.csv'
def update_dashboard():
    # (Dashboard code is the same)
    st.sidebar.write("### Total Debris Detected So Far:")
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if not df.empty:
            detection_counts = df['detected_class'].value_counts()
            fig, ax = plt.subplots(); ax.bar(detection_counts.index, detection_counts.values, color='#00A9FF'); ax.set_ylabel('Total Count'); ax.tick_params(axis='x', rotation=45); ax.set_ylim(bottom=0); fig.patch.set_alpha(0); ax.patch.set_alpha(0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white'); ax.yaxis.label.set_color('white'); st.sidebar.pyplot(fig, use_container_width=True)
            with st.sidebar.expander("Show Detailed Detection History"): st.dataframe(df.tail(10))
        else: st.sidebar.info("No detections recorded yet.")
    else: st.sidebar.info("No detections recorded yet.")
update_dashboard()

# MAIN PAGE CONTENT
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 8px #000;'>🌊 Marine Debris Detection Agent</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white; text-shadow: 2px 2px 8px #000;'>'Save the Ocean, One Image at a Time'</h2>", unsafe_allow_html=True)
st.write(" "); st.divider()

uploaded_file = st.file_uploader("Upload an underwater image to detect debris...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    if model is not None:
        with st.spinner('AI is analyzing the image...'):
            img_array = np.array(image)
            results = model.predict(source=img_array)
            result_image_array = results[0].plot()
            result_image_rgb = cv2.cvtColor(result_image_array, cv2.COLOR_BGR2RGB)
        
        # (The rest of the display logic is the same)
        st.divider(); col1, col2 = st.columns(2)
        with col1: st.write("### Original Image"); st.image(image, use_container_width=True)
        with col2: st.write("### AI Detection Results"); st.image(result_image_rgb, use_container_width=True)
        st.divider(); st.write("### 📝 Detection Summary")
        boxes = results[0].boxes
        if len(boxes) == 0: st.success("✅ No debris detected!")
        else:
            detections = []; report_str = f"Detection Report for {uploaded_file.name}\n" + f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "-------------------------\n"; detected_classes = []
            for box in boxes:
                class_id = int(box.cls); class_name = model.names[class_id]; confidence = float(box.conf); detections.append({'class': class_name, 'confidence': confidence})
                line = f"- Found **{class_name}** with **{int(confidence*100)}%** confidence."; st.write(line); report_str += f"Found {class_name} with {int(confidence*100)}% confidence.\n"; detected_classes.append(class_name)
            try:
                new_data = pd.DataFrame({'detected_class': detected_classes})
                if not os.path.exists(HISTORY_FILE): new_data.to_csv(HISTORY_FILE, index=False)
                else: new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
            except Exception as e: st.error(f"Could not save detection history: {e}")
            st.divider(); dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1: st.download_button(label="📥 Download Report", data=report_str, file_name=f"report_{uploaded_file.name}.txt", mime="text/plain")
            with dl_col2: result_img_pil = Image.fromarray(result_image_rgb); buf = io.BytesIO(); result_img_pil.save(buf, format="PNG"); byte_im = buf.getvalue(); st.download_button(label="🖼️ Download Image", data=byte_im, file_name=f"detected_{uploaded_file.name}.png", mime="image/png")
            detection_counts = Counter(detected_classes); fig, ax = plt.subplots(); ax.bar(detection_counts.keys(), detection_counts.values()); ax.set_ylabel('Count'); ax.set_title('Debris Detection Count'); plt.xticks(rotation=45, ha='right'); st.write("### 📊 Detection Chart"); st.pyplot(fig)
            chart_buf = io.BytesIO(); fig.savefig(chart_buf, format="png", bbox_inches='tight')
            with dl_col3: st.download_button(label="📈 Download Chart", data=chart_buf, file_name=f"chart_{uploaded_file.name}.png", mime="image/png")
    else: st.warning("Model is not available.")
st.divider()
with st.expander("About This Project"): st.info("**What it is:** The Marine Debris Detection Agent is an AI-powered tool to combat ocean pollution...\n\n**Technology Used:** Python, Streamlit, PyTorch, Ultralytics (YOLOv8), Pandas, OpenCV.")
