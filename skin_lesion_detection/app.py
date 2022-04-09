### Script for Cloud Computing Skin Lesion Project
# Part of code borrowed from: https://github.com/mrdbourke/cs329s-ml-deployment-tutorial 

import os
import json
import requests
import numpy as np
import SessionState
import streamlit as st
import tensorflow as tf
from utils import get_image, load_and_prep_image, classes_and_models, update_logger, predict_json

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ccproject-343606-498535894d22.json" # change for your GCP key
PROJECT = "ccproject-343606" # change for your GCP project
REGION = None # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to Skin Lesion Detection in Clinical Image")
st.header("Detect suspicious skin lesions in you image")

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image = get_image(image)
    # # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    # image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    # # image = tf.expand_dims(image, axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    
    preds = preds[-1]['sequential']
    pred_class = class_names[np.argmax(preds)]

    return pred_class

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (6 classes)", # original 6 classes
     "Model 2 (Ugly duckling)", # ugly duckling model
    )
)

# Model choice logic
if choose_model == "Model 1 (6 classes)":
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]
elif choose_model == "Model 2 (Ugly duckling)":
    CLASSES = classes_and_models["model_2"]["classes"]
    MODEL = classes_and_models["model_2"]["model_name"]

# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose {MODEL}, these are the classes of patterns it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of food",
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    # session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    prediction = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {prediction}")
    # st.write(f"Prediction: {prediction}, \
    #            Confidence: {session_state.pred_conf:.3f}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=session_state.image,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct label be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()