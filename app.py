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

# =================================================================================
# 1. STYLING AND PAGE CONFIGURATION
# =================================================================================

def set_bg_from_local(image_file):
    """
    Sets a background image from a local file using base64 encoding.
    """
    try:
        with open(image_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-attachment: fixed;
                background-size: cover;
            }}
            /* CSS for mobile responsiveness */
            @media (max-width: 640px) {{
                h1 {{ font-size: 2.2rem !important; }}
                h2 {{ font-size: 1.5rem !important; }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Background image '{image_file}' not found. Make sure it's in the same folder as app.py.")

# Set the page layout to centered and apply the background
st.set_page_config(page_title="Marine Debris Detector", page_icon="🌊", layout="centered")
set_bg_from_local("bg.jpg")

# =================================================================================
# 2. MODEL LOADING
# =================================================================================

@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLO model from the specified path and caches it for efficiency.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'best.pt' is in the same folder as 'app.py'.")
        return None

# Load the trained AI model
model = load_model('best.pt')

# =================================================================================
# 3. SIDEBAR AND DASHBOARD
# =================================================================================

st.sidebar.title("📊 Analytics Dashboard")
HISTORY_FILE = 'detection_history.csv'

def update_dashboard():
    """
    Reads the detection history and displays charts and data in the sidebar.
    """
    st.sidebar.write("### Total Debris Detected So Far:")
    
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if not df.empty:
            detection_counts = df['detected_class'].value_counts()
            
            # Create a Matplotlib chart for the dashboard
            fig, ax = plt.subplots()
            ax.bar(detection_counts.index, detection_counts.values, color='#00A9FF')
            ax.set_ylabel('Total Count')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(bottom=0)
            
            # Style the chart for Streamlit's dark theme
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.yaxis.label.set_color('white')
            
            st.sidebar.pyplot(fig, use_container_width=True)
            
            # Show detailed history in a collapsible expander
            with st.sidebar.expander("Show Detailed Detection History"):
                st.dataframe(df.tail(10))
        else:
            st.sidebar.info("No detections recorded yet.")
    else:
        st.sidebar.info("No detections recorded yet.")

# Display the dashboard when the app first loads
update_dashboard()

# =================================================================================
# 4. MAIN PAGE CONTENT
# =================================================================================

# Display the main title and slogan using centered markdown
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 8px #000;'>🌊 Marine Debris Detection Agent</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white; text-shadow: 2px 2px 8px #000;'>'Save the Ocean, One Image at a Time'</h2>", unsafe_allow_html=True)
st.write(" ") 
st.divider()

# Create the file uploader for user input
uploaded_file = st.file_uploader("Upload an underwater image to detect debris...", type=["jpg", "jpeg", "png"])

# Process the uploaded file
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    if model is not None:
        # Show a spinner while the model performs inference
        with st.spinner('AI is analyzing the image... Please wait.'):
            img_array = np.array(image)
            results = model.predict(source=img_array)
            result_image_array = results[0].plot()
            result_image_rgb = cv2.cvtColor(result_image_array, cv2.COLOR_BGR2RGB)

        # Display original and detected images side-by-side
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.write("### AI Detection Results")
            st.image(result_image_rgb, use_container_width=True)
        
        # Display the detailed text summary of detections
        st.divider()
        st.write("### 📝 Detection Summary")
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.success("✅ No debris detected in the image!")
        else:
            detections = []
            for box in boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                detections.append({'class': class_name, 'confidence': confidence})

            # Prepare the text report for display and download
            report_str = f"Detection Report for {uploaded_file.name}\n"
            report_str += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_str += "-------------------------\n"
            
            detected_classes = []
            for det in detections:
                line = f"- Found **{det['class']}** with **{int(det['confidence']*100)}%** confidence."
                st.write(line)
                report_str += f"Found {det['class']} with {int(det['confidence']*100)}% confidence.\n"
                detected_classes.append(det['class'])
            
            # Save the new detections to the history file for the dashboard
            try:
                new_data = pd.DataFrame({'detected_class': detected_classes})
                if not os.path.exists(HISTORY_FILE):
                    new_data.to_csv(HISTORY_FILE, index=False)
                else:
                    new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
            except Exception as e:
                st.error(f"Could not save detection history: {e}")

            st.divider()

            # Create download buttons for the report, image, and chart
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(label="📥 Download Report (.txt)", data=report_str, file_name=f"report_{uploaded_file.name}.txt", mime="text/plain")
            with dl_col2:
                result_img_pil = Image.fromarray(result_image_rgb)
                buf = io.BytesIO()
                result_img_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(label="🖼️ Download Image", data=byte_im, file_name=f"detected_{uploaded_file.name}.png", mime="image/png")
            
            # Create and display the per-image bar chart
            detection_counts = Counter(detected_classes)
            fig, ax = plt.subplots()
            ax.bar(detection_counts.keys(), detection_counts.values())
            ax.set_ylabel('Count')
            ax.set_title('Debris Detection Count')
            plt.xticks(rotation=45, ha='right')
            
            st.write("### 📊 Detection Chart")
            st.pyplot(fig)

            # Prepare the chart for download
            chart_buf = io.BytesIO()
            fig.savefig(chart_buf, format="png", bbox_inches='tight')
            with dl_col3:
                st.download_button(label="📈 Download Chart", data=chart_buf, file_name=f"chart_{uploaded_file.name}.png", mime="image/png")
            
    elif model is None:
        st.warning("Please wait for the AI model to be downloaded and loaded.")

# Fallback message if model loading fails at the start
elif model is None:
    st.warning("Model could not be loaded. Please ensure 'best.pt' is available and restart the app.")

# =================================================================================
# 5. ABOUT SECTION
# =================================================================================
st.divider()
with st.expander("About This Project"):
    st.info(
        "**What it is:** The Marine Debris Detection Agent is an AI-powered tool to combat ocean pollution. "
        "It uses a YOLOv8 model to detect and classify 15 types of underwater debris from images.\n\n"
        "**Technology Used:** Python, Streamlit, PyTorch, Ultralytics (YOLOv8), Pandas, OpenCV."
    )