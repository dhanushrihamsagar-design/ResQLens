import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from detector import detect_objects, load_detector
from risk_engine import assess_risk


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResQLens",
    page_icon="🚨",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_detector()


# --------------------------------------------------
# DRAW DETECTIONS
# --------------------------------------------------

def draw_detections(image, detections):

    annotated = image.convert("RGB").copy()

    draw = ImageDraw.Draw(annotated)

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        class_name = detection["class"]
        confidence = detection["confidence"]

        # Bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        # Label
        label = f"{class_name} {confidence:.0%}"

        draw.text(
            (x1, max(0, y1 - 20)),
            label,
            fill="red"
        )

    return annotated


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚨 ResQLens")

st.subheader(
    "Offline AI Disaster Vision & Hazard Prioritization"
)

st.write(
    "Upload a disaster image to detect objects locally "
    "and identify potential hazards using a rule-based "
    "risk reasoning layer."
)

st.divider()


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI Model", "YOLOv8-N")

with col2:
    st.metric("Input", "640 × 640")

with col3:
    st.metric("Processing", "Local / Offline")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a disaster image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    # --------------------------------------------------
    # SCENE CONTEXT
    # --------------------------------------------------

    scene_context = st.selectbox(
        "Select scene context",
        [
            "Flooded road",
            "Urban disaster",
            "General disaster scene"
        ]
    )

    # --------------------------------------------------
    # RUN DETECTION
    # --------------------------------------------------

    with st.spinner("Running YOLOv8-N detection..."):

        model = get_model()

        detections = detect_objects(
            image,
            confidence_threshold=0.35,
            model=model
        )

    # --------------------------------------------------
    # ANNOTATED IMAGE
    # --------------------------------------------------

    annotated_image = draw_detections(
        image,
        detections
    )

    st.subheader("🔍 Detection Result")

    st.image(
        annotated_image,
        caption="YOLOv8-N detected objects",
        use_container_width=True
    )

    # --------------------------------------------------
    # DETECTION SUMMARY
    # --------------------------------------------------

    st.subheader("📊 Detected Objects")

    if detections:

        for detection in detections:

            st.write(
                f"**{detection['class'].title()}** — "
                f"{detection['confidence']:.0%} confidence"
            )

    else:

        st.info("No objects detected above the confidence threshold.")

    # --------------------------------------------------
    # RISK ANALYSIS
    # --------------------------------------------------

    st.divider()

    st.subheader("⚠️ Hazard Prioritization")

    result = assess_risk(detections)

    risk_level = result["level"]

    if risk_level == "HIGH":
        st.error(f"🚨 Risk Level: {risk_level}")

    elif risk_level == "MEDIUM":
        st.warning(f"⚠️ Risk Level: {risk_level}")

    else:
        st.success(f"✅ Risk Level: {risk_level}")

    st.write(
        f"**Scene:** {scene_context}"
    )

    st.write(
        f"**Reason:** {result['reason']}"
    )

    st.write(
        f"**Recommended Action:** {result['action']}"
    )

    # --------------------------------------------------
    # TECHNICAL TRANSPARENCY
    # --------------------------------------------------

    with st.expander("ℹ️ How ResQLens works"):

        st.write(
            "1. The uploaded image is processed locally."
        )

        st.write(
            "2. Qualcomm AI Hub YOLOv8-N performs object detection."
        )

        st.write(
            "3. Confidence filtering and Non-Maximum Suppression "
            "remove duplicate detections."
        )

        st.write(
            "4. ResQLens applies a separate risk-reasoning layer "
            "to the detected objects."
        )

        st.write(
            "5. The system produces a potential hazard level, "
            "explanation, and recommended safety action."
        )

        st.info(
            "YOLOv8-N detects objects such as cars and people. "
            "It does not directly detect 'flood risk'. "
            "ResQLens uses scene context and detected objects "
            "to estimate potential hazards."
        )