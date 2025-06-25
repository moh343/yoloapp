import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_HUB"] = "0"
os.environ["ULTRALYTICS_NO_HUB"] = "1"
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np



# 🧠 Modell laden
try:
    model = model = YOLO("runs\detect\train\weights\beset.pt")# Kleines YOLOv8-Modell (wird automatisch heruntergeladen)
except Exception as e:
    st.error(f"Fehler beim Laden des YOLO-Modells: {e}")
    st.stop()

# UI
st.title("🧠 YOLOv8 Objekterkennung")
st.write("Lade ein Bild hoch, um Objekte automatisch erkennen zu lassen.")

# Upload-Feld
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png"])


# st.set_page_config(page_title="Test", layout="centered", initial_sidebar_state="expanded")

# st.title("✅ Streamlit funktioniert!")
# st.write("Wenn du das siehst, ist alles korrekt installiert.")
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Hochgeladenes Bild", use_column_width=True)

        image_np = np.array(image)

        with st.spinner("YOLOv8 verarbeitet das Bild..."):
            results = model(image_np)[0]

        st.write(f"Gefundene Objekte: {len(results.boxes)}")

        # Convert and show annotated image
        annotated = results.plot()
        annotated_image = Image.fromarray(annotated)
        st.image(annotated_image, caption="🔍 Erkannte Objekte", use_column_width=True)

        # List detected objects
        st.subheader("📋 Erkannte Objekte:")
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.model.names[class_id]
            st.write(f"- **{class_name}** ({confidence:.1%})")

    except Exception as e:
        st.error(f"❌ Fehler bei der Verarbeitung des Bildes: {e}")