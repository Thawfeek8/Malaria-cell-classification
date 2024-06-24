import streamlit as st
import pandas as pd
import time
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
model = load_model('malaria_detector.h5')

# Set background color for the entire app
st.markdown(
    """
    <style>
        body {
            background-color: #000000;
            color: #ffffff;
        }
        .st-bd {
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define the Streamlit app
def main():
    
    st.title("Malaria Detection App")
    st.write(
        "This app uses a deep learning model to predict whether a given cell image is parasitized or uninfected."
    )
    page = st.sidebar.selectbox("Select Page", ["Predict", "Explore"])

    if page == "Predict":
        predict_page()
    elif page == "Explore":
        explore_page()

def predict_page():
     #  model info
    st.header("Model Information")
    st.text("Model Architecture: Convolutional Neural Network")
    st.text("Framework: TensorFlow")
    st.text("Dataset: Malaria Cell Images")  
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        with st.spinner("Processing..."):
            time.sleep(2)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        image = image.resize((130, 130))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)[0][0]

        

        # Display the prediction
        st.write("Prediction Probability:", prediction)
        if prediction > 0.5:
            st.success("Result: Uninfected")
            st.write('Explanation: The model predicts that the image contains a uninfected cell. This suggests the absence of malaria.')
        else:
            st.error("Result: Parasitized")
            st.write('Explanation: The model predicts that the image contains a parasitized cell. This may indicate the presence of malaria.')


# Function to plot line chart (change this according to your specific data)
def plot_line_chart():
    epochs = np.arange(1, 21)
    accuracy = np.random.rand(20) * 0.9 + 0.5  # Random accuracies between 0.5 and 1

    # Create the line chart
    st.line_chart(data={"Epoch": epochs, "Accuracy": accuracy},color=["#FF0000", "#0000FF"])
   

      
def explore_page():
     st.header("Explore Page")
     total_images=12400+5900
     parasitized_count=12400
     uninfected_count=5900


     
    
     # Display dataset information
     st.subheader("Dataset Information")
     st.write(f"Number of Images: {total_images}")
     st.write(f"Number of Parasitized Cells: {parasitized_count}")
     st.write(f"Number of Uninfected Cells: {uninfected_count}")

     
     # Prediction Insights
     st.header("Prediction Insights")


     st.write("It's important to note that the model may face challenges in cases where:")
     st.write("- The image quality is poor or inconsistent.")
     st.write("- Cells exhibit variations not well-represented in the training set.")
     st.write("- Other factors affecting image clarity or content.")


     st.write("Interpreting predictions:")
     st.write("- Probabilities closer to 0 indicate a high confidence in the 'Uninfected' class.")
     st.write("- Probabilities closer to 1 indicate a high confidence in the 'Parasitized' class.")
    


     # Plot line chart
     st.subheader("Model performance")
     plot_line_chart()
     
     st.subheader("Count of Parasitized and Uninfected images")
     chart_data = pd.DataFrame([[12400,5900]], columns=["Parasitized","Uninfected"])
     st.bar_chart(chart_data)


# Run the app
if __name__ == "__main__":
    main()








