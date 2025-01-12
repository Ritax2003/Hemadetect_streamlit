import streamlit as st
import torch
from torchvision.models import googlenet, GoogLeNet_Weights
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Function to initialize the model
def initialize_model():
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(in_features=1024, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=64, out_features=4)
    )
    model.load_state_dict(torch.load(r'blood_cancer_model.pth'))
    model.eval()
    return model

# Transformation function
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img



# Streamlit app
st.title("HemaDetect - Blood Cancer Prediction")
st.write("Upload an image to predict the type of blood cancer.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load model
    model = initialize_model()

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_tensor = preprocess_image(uploaded_file)

    # Select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    img_tensor = img_tensor.to(device)



    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Labels map
    labels_map = {0: 'Benign', 1: 'Early_Pre_B', 2: 'Pre_B', 3: 'Pro_B'}
    predicted_class_name = labels_map[predicted_class.item()]
    confidence_score = confidence.item()


    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class_name}")
    st.write(f"**Confidence Score:** {confidence_score:.4f}")

  
