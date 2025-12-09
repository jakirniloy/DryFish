# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import pandas as pd
# import plotly.express as px

# # -------------------------------
# # Streamlit config
# # -------------------------------
# st.set_page_config(
#     page_title="Dry Fish Classifier",
#     layout="wide",
#     page_icon="🐟",
# )

# # -------------------------------
# # Custom CSS for professional layout
# # -------------------------------
# st.markdown(
#     """
#     <style>
#     /* Main page background */
#     .stApp {
#         background-color: #e0f7fa;  /* light blue */
#         color: #000000;
#         font-family: 'Arial', sans-serif;
#     }

#     /* Sidebar style */
#     .css-1d391kg {
#         background-color: #0288d1;  /* blue */
#         color: white;
#     }

#     /* Sidebar header */
#     .css-1v0mbdj h2 {
#         color: white;
#     }

#     /* Card style for outputs */
#     .card {
#         background-color: white;
#         border-radius: 15px;
#         padding: 20px;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.2);
#         margin-bottom: 20px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.title("🐟 Dry Fish Classification with Grad-CAM")

# # -------------------------------
# # Load Model
# # -------------------------------
# @st.cache_resource
# def load_trained_model():
#     return load_model("hybrid_cnn_attention_dryfish.h5")

# model = load_trained_model()
# model.trainable = False

# # -------------------------------
# # Class labels and descriptions
# # -------------------------------
# class_names = [
#     'Bashpata', 'Chanda', 'Chapila', 'Chewa', 'Churi', 'Loitta',
#     'Shukna Feuwa', 'Shundori', 'chingri', 'kachki', 'narkeli', 'puti chepa'
# ]

# class_descriptions = {
#     'Bashpata': "Flat-bodied dried fish with elongated shape, typically sun-dried.",
#     'Chanda': "Small, oval-shaped fish known for its translucent body texture.",
#     'Chapila': "Medium-sized dried fish with silver scales, common in river catch.",
#     'Chewa': "Slender-bodied fish, popular in coastal markets.",
#     'Churi': "Long, ribbon-like fish with sharp head features.",
#     'Loitta': "Cylindrical-bodied fish with distinctive dorsal fin pattern.",
#     'Shukna Feuwa': "Curved-bodied dried fish, often heavily salted during processing.",
#     'Shundori': "Small, slender fish with delicate body structure.",
#     'chingri': "Dried shrimp, small, used widely in curries and condiments.",
#     'kachki': "Small fish species with narrow bodies, usually sun-dried.",
#     'narkeli': "Medium-sized dried fish with firm body texture.",
#     'puti chepa': "Flat-bodied dried fish with noticeable fin edges."
# }

# # -------------------------------
# # Preprocess image
# # -------------------------------
# def preprocess(img):
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     return img_array

# # -------------------------------
# # Manual Grad-CAM
# # -------------------------------
# def manual_gradcam(img_array, input_img):
#     last_conv_layer = None
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             last_conv_layer = layer
#             break
#     if last_conv_layer is None:
#         return input_img

#     conv_model = Model(inputs=model.inputs, outputs=last_conv_layer.output)
#     conv_output = conv_model.predict(img_array)[0]

#     heatmap = np.mean(conv_output, axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap) + 1e-10

#     heatmap = cv2.resize(heatmap, (input_img.width, input_img.height))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     overlay = cv2.addWeighted(np.array(input_img), 0.6, heatmap, 0.4, 0)
#     return overlay

# # -------------------------------
# # Sidebar for upload & buttons
# # -------------------------------
# st.sidebar.header("Upload Image")
# uploaded_file = st.sidebar.file_uploader("Choose a fish image...", type=["jpg","jpeg","png"])
# predict_btn = st.sidebar.button("Predict Class")
# gradcam_btn = st.sidebar.button("Show Grad-CAM")

# # -------------------------------
# # Main display logic
# # -------------------------------
# if uploaded_file:
#     input_img = Image.open(uploaded_file).convert("RGB")
#     img_array = preprocess(input_img)

#     # -------------------
#     # Prediction button
#     # -------------------
#     if predict_btn:
#         preds = model.predict(img_array)
#         class_index = int(np.argmax(preds))
#         predicted_class = class_names[class_index]

#         # Prediction & description
       
#         st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown("### Uploded Image")
#         resized_input = input_img.resize((200, 200))
#         st.image(resized_input)
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown(f"### 🎯 Predicted Class: **{predicted_class}**")
#         st.markdown(f"### 🔹 Confidence: **{preds[0][class_index]*100:.2f}%**")
#         st.markdown(f"### 📖 Local Description:")
#         st.write(class_descriptions[predicted_class])
#         st.markdown('</div>', unsafe_allow_html=True)

#         # -------------------
#         # Prediction Probabilities Chart
#         # -------------------
#         prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
#         prob_df = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Probability'])
#         prob_df['Probability (%)'] = prob_df['Probability'] * 100

#         fig = px.bar(prob_df, x='Class', y='Probability (%)', 
#                      color='Probability (%)', 
#                      color_continuous_scale='blues',
#                      text='Probability (%)',
#                      title="📊 Prediction Probabilities")
#         fig.update_layout(xaxis_title="Class", yaxis_title="Probability (%)", 
#                           uniformtext_minsize=8, uniformtext_mode='hide')
#         st.plotly_chart(fig, use_container_width=True)

#     # -------------------
#     # Grad-CAM button
#     # -------------------
#     if gradcam_btn:
#         gradcam_overlay = manual_gradcam(img_array, input_img)

#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown('<div class="card">', unsafe_allow_html=True)
#             st.markdown("### Original Image")
#             resized_input = input_img
#             st.image(resized_input)
#             st.markdown('</div>', unsafe_allow_html=True)

#         with col2:
#             st.markdown('<div class="card">', unsafe_allow_html=True)
#             st.markdown("### Grad-CAM Overlay")
#             resized_gradcam = gradcam_overlay
#             st.image(resized_gradcam) 
#             st.markdown('</div>', unsafe_allow_html=True)
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd

# -------------------------------
# Streamlit config
# -------------------------------
st.set_page_config(
    page_title="Dry Fish Classifier",
    layout="wide",
    page_icon="🐟",
)

# -------------------------------
# Custom CSS for professional layout
# -------------------------------
st.markdown(
    """
    <style>
    /* Main page background */
    .stApp {
        background-color: #e0f7fa;  /* light blue */
        color: #000000;
        font-family: 'Arial', sans-serif;
    }

    /* Sidebar style */
    .css-1d391kg {
        background-color: #0288d1;  /* blue */
        color: white;
    }

    /* Sidebar header */
    .css-1v0mbdj h2 {
        color: white;
    }

    /* Card style for outputs */
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐟 Dry Fish Classification with Grad-CAM")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("hybrid_cnn_attention_dryfish.h5")

model = load_trained_model()
model.trainable = False

# -------------------------------
# Class labels and descriptions
# -------------------------------
class_names = [
    'Bashpata', 'Chanda', 'Chapila', 'Chewa', 'Churi', 'Loitta',
    'Shukna Feuwa', 'Shundori', 'chingri', 'kachki', 'narkeli', 'puti chepa'
]

class_descriptions = {
    'Bashpata': "Flat-bodied dried fish with elongated shape, typically sun-dried.",
    'Chanda': "Small, oval-shaped fish known for its translucent body texture.",
    'Chapila': "Medium-sized dried fish with silver scales, common in river catch.",
    'Chewa': "Slender-bodied fish, popular in coastal markets.",
    'Churi': "Long, ribbon-like fish with sharp head features.",
    'Loitta': "Cylindrical-bodied fish with distinctive dorsal fin pattern.",
    'Shukna Feuwa': "Curved-bodied dried fish, often heavily salted during processing.",
    'Shundori': "Small, slender fish with delicate body structure.",
    'chingri': "Dried shrimp, small, used widely in curries and condiments.",
    'kachki': "Small fish species with narrow bodies, usually sun-dried.",
    'narkeli': "Medium-sized dried fish with firm body texture.",
    'puti chepa': "Flat-bodied dried fish with noticeable fin edges."
}

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# -------------------------------
# Manual Grad-CAM
# -------------------------------
def manual_gradcam(img_array, input_img):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        return input_img

    conv_model = Model(inputs=model.inputs, outputs=last_conv_layer.output)
    conv_output = conv_model.predict(img_array)[0]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    heatmap = cv2.resize(heatmap, (input_img.width, input_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(np.array(input_img), 0.6, heatmap, 0.4, 0)
    return overlay

# -------------------------------
# Sidebar for upload & buttons
# -------------------------------
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a fish image...", type=["jpg","jpeg","png"])
predict_btn = st.sidebar.button("Predict Class")
gradcam_btn = st.sidebar.button("Show Grad-CAM")

# -------------------------------
# Main display logic
# -------------------------------
if uploaded_file:
    input_img = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess(input_img)

    # -------------------
    # Prediction button
    # -------------------
    if predict_btn:
        preds = model.predict(img_array)
        class_index = int(np.argmax(preds))
        predicted_class = class_names[class_index]
        confidence = preds[0][class_index]

        if confidence < 0.5:
             st.markdown("### Uploded Image")
             resized_input = input_img.resize((200, 200))
             st.image(resized_input)
             st.warning("⚠️ The uploaded image does not appear to be a recognized fish.")
        else:
            # Prediction & description
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Uploded Image")
            resized_input = input_img.resize((200, 200))
            st.image(resized_input)
            st.markdown(f"### 🎯 Predicted Class: **{predicted_class}**")
            st.markdown(f"### 🔹 Confidence: **{confidence*100:.2f}%**")
            st.markdown(f"### 📖 Local Description:")
            st.write(class_descriptions[predicted_class])
            st.markdown('</div>', unsafe_allow_html=True)

            # Prediction Probabilities Chart
            prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
            prob_df = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Probability'])
            prob_df['Probability (%)'] = prob_df['Probability'] * 100
            st.bar_chart(prob_df.set_index('Class')['Probability (%)'])

    # -------------------
    # Grad-CAM button
    # -------------------
    if gradcam_btn:
        

        gradcam_overlay = manual_gradcam(img_array, input_img)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Original Image")
            resized_input = input_img
            st.image(resized_input)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Grad-CAM Overlay")
            resized_gradcam = gradcam_overlay
            st.image(resized_gradcam) 
            st.markdown('</div>', unsafe_allow_html=True)
