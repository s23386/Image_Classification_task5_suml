# Import necessary libraries
import streamlit as st
from fastai.vision.all import *

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return load_learner("s23386_image_model.pkl")

# Define Streamlit app
def main():
    # Load the model
    model = load_model()
    
    # Set title and description
    st.title("Dog or Cat Classifier")
    st.write("Upload an image and we'll classify it as either a dog or a cat.")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        img = PILImage.create(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        pred, pred_idx, probs = model.predict(img)
        
        # Display the prediction
        st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.4f}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
