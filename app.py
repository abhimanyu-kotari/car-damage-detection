
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import io

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoInspect AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 12px;
    }

    .metric-card {
        background-color: #151e2e;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #38bdf8;
        font-size: 28px;
        font-weight: 800;
    }

    .damage-card {
        background-color: #151e2e;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .damage-name {
        color: #f1f5f9;
        font-size: 16px;
        font-weight: 700;
    }

    .damage-confidence {
        color: #38bdf8;
        font-size: 14px;
        font-weight: 600;
    }

    div.stButton > button {
        background-color: #0ea5e9;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #0284c7;
        color: white;
    }

    .stImage img {
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD TRAINED YOLO MODEL
# =========================================================

@st.cache_resource
def load_model(model_path="best.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        if os.path.exists("best_nano.pt"):
            return YOLO("best_nano.pt")
        raise e

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🚗 AutoInspect AI")

    st.caption("Intelligent Vehicle Damage Analysis")

    st.divider()

    model_options = ["YOLOv8 Medium (High Accuracy)", "YOLOv8 Nano (Fast / Fallback)"] if os.path.exists("best_nano.pt") else ["YOLOv8 Medium"]
    selected_model_label = st.selectbox(
        "Model Version",
        options=model_options,
        index=0,
        help="Select between the high-accuracy Medium model and the lightweight Nano model."
    )
    selected_model_path = "best_nano.pt" if "Nano" in selected_model_label else "best.pt"
    model = load_model(selected_model_path)

    st.markdown("### Model Information")

    st.write(f"**Architecture:** {selected_model_label.split(' (')[0]} Segmentation")
    st.write("**Model:** Custom Trained")
    st.write("**Damage Classes:** 6")

    st.divider()

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.caption(
        "Higher confidence values show fewer, "
        "more confident detections."
    )

    st.divider()

    st.caption(
        "Built with Python, Streamlit and YOLOv8"
    )

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Vehicle Damage Inspection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a vehicle image and let AI identify damaged areas.'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of the damaged vehicle."
)

if uploaded_file is None:

    st.info(
        "👆 Upload a car image to begin the damage inspection."
    )

else:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        '<div class="section-title">Image Preview</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded Vehicle Image",
        use_container_width=True
    )

    detect_button = st.button(
        "🔍 Analyze Vehicle Damage",
        type="primary"
    )

    # =====================================================
    # RUN MODEL PREDICTION
    # =====================================================

    if detect_button:

        temp_path = None

        try:

            with st.spinner("AI is analyzing the vehicle..."):

                # Save uploaded image temporarily
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False
                ) as temp_file:

                    image.save(temp_file.name)
                    temp_path = temp_file.name

                # Run YOLO prediction
                results = model.predict(
                    source=temp_path,
                    conf=confidence,
                    imgsz=640,
                    save=False
                )

                result = results[0]

                # Generate annotated image
                annotated_image = result.plot()

                # Convert BGR to RGB
                annotated_image = annotated_image[:, :, ::-1]

                # Convert to PIL Image
                annotated_image = Image.fromarray(
                    annotated_image
                )

                # Extract detections
                detections = []

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(box.cls[0])

                        class_name = model.names[class_id]

                        conf = float(box.conf[0])

                        detections.append({
                            "name": class_name,
                            "confidence": conf
                        })

            # =================================================
            # INSPECTION RESULTS
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Inspection Results'
                '</div>',
                unsafe_allow_html=True
            )

            # =================================================
            # METRICS
            # =================================================

            detection_count = len(detections)

            average_confidence = (
                np.mean([
                    detection["confidence"]
                    for detection in detections
                ])
                if detections else 0
            )

            unique_classes = len(
                set(
                    detection["name"]
                    for detection in detections
                )
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Detected Damages
                        </div>

                        <div class="metric-value">
                            {detection_count}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Average Confidence
                        </div>

                        <div class="metric-value">
                            {average_confidence:.0%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Damage Categories
                        </div>

                        <div class="metric-value">
                            {unique_classes}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # ORIGINAL VS DETECTED IMAGE
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Original vs AI Detection'
                '</div>',
                unsafe_allow_html=True
            )

            original_col, result_col = st.columns(2)

            with original_col:

                st.markdown("**Original Image**")

                st.image(
                    image,
                    use_container_width=True
                )

            with result_col:

                st.markdown("**AI Segmentation**")

                st.image(
                    annotated_image,
                    use_container_width=True
                )

            # =================================================
            # DAMAGE DETAILS
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Damage Details'
                '</div>',
                unsafe_allow_html=True
            )

            if detections:

                for index, detection in enumerate(
                    detections,
                    start=1
                ):

                    damage_name = (
                        detection["name"]
                        .replace("_", " ")
                        .title()
                    )

                    damage_confidence = (
                        detection["confidence"]
                    )

                    st.markdown(
                        f"""
                        <div class="damage-card">

                            <div class="damage-name">
                                {index}. {damage_name}
                            </div>

                            <div class="damage-confidence">
                                Confidence:
                                {damage_confidence:.1%}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.warning(
                    "No damage was detected. "
                    "Try lowering the confidence threshold "
                    "or uploading another image."
                )

            # =================================================
            # DOWNLOAD ANNOTATED IMAGE
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Download Inspection Image'
                '</div>',
                unsafe_allow_html=True
            )

            # Convert annotated image into PNG bytes
            image_buffer = io.BytesIO()

            annotated_image.save(
                image_buffer,
                format="PNG"
            )

            image_bytes = image_buffer.getvalue()

            st.download_button(
                label="⬇️ Download Annotated Image",
                data=image_bytes,
                file_name="car_damage_result.png",
                mime="image/png"
            )

        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as error:

            st.error(
                f"An error occurred during prediction: {error}"
            )

        # =====================================================
        # CLEAN TEMPORARY FILE
        # =====================================================

        finally:

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):

                os.remove(temp_path)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AutoInspect AI | AI-assisted vehicle damage detection. "
    "Predictions are estimates and should be reviewed "
    "by a qualified vehicle inspector."
)
