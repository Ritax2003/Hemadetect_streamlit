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

# Resource usage functions
def get_cpu_memory_usage():
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Convert bytes to MB
    return cpu_usage, memory_usage

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert bytes to MB
        return gpu_memory_allocated, gpu_memory_reserved
    return None, None

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

    # Get resource usage before prediction
    cpu_usage_before, memory_usage_before = get_cpu_memory_usage()
    gpu_memory_allocated_before, gpu_memory_reserved_before = get_gpu_memory_usage()

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Labels map
    labels_map = {0: 'Benign', 1: 'Early_Pre_B', 2: 'Pre_B', 3: 'Pro_B'}
    predicted_class_name = labels_map[predicted_class.item()]
    confidence_score = confidence.item()

    # Get resource usage after prediction
    cpu_usage_after, memory_usage_after = get_cpu_memory_usage()
    gpu_memory_allocated_after, gpu_memory_reserved_after = get_gpu_memory_usage()

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class_name}")
    st.write(f"**Confidence Score:** {confidence_score:.4f}")

    # st.subheader("Resource Usage")
    # st.write(f"**CPU Usage Before:** {cpu_usage_before:.2f}%")
    # st.write(f"**Memory Usage Before:** {memory_usage_before:.2f} MB")
    # st.write(f"**CPU Usage After:** {cpu_usage_after:.2f}%")
    # st.write(f"**Memory Usage After:** {memory_usage_after:.2f} MB")
    # st.write(f"**CPU Used:** {cpu_usage_after - cpu_usage_before:.2f}%")
    # st.write(f"**Memory Used:** {memory_usage_after - memory_usage_before:.2f} MB")

    # if gpu_memory_allocated_before is not None and gpu_memory_reserved_before is not None:
    #     st.write(f"**GPU Memory Allocated Before:** {gpu_memory_allocated_before:.2f} MB")
    #     st.write(f"**GPU Memory Reserved Before:** {gpu_memory_reserved_before:.2f} MB")
    #     st.write(f"**GPU Memory Allocated After:** {gpu_memory_allocated_after:.2f} MB")
    #     st.write(f"**GPU Memory Reserved After:** {gpu_memory_reserved_after:.2f} MB")
    #     st.write(f"**GPU Memory Allocated:** {gpu_memory_allocated_after - gpu_memory_allocated_before:.2f} MB")
    #     st.write(f"**GPU Memory Reserved:** {gpu_memory_reserved_after - gpu_memory_reserved_before:.2f} MB")
