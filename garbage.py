import streamlit as st
import numpy as np
import cv2
import os
import torch
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

# Define the ResNet model class
class ResNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=False)  # Load the model without pretrained weights

        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)  # Define the new classification layer

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

# Create an instance of the ResNet model
loaded_model = torch.load("garbage_model.pt", map_location=torch.device('cpu'))
model = ResNet(num_classes=12)  # Adjust the number of classes accordingly

# Load the trained weights
model.load_state_dict(loaded_model.state_dict())
model.eval()

# Define the class labels
class_labels = ['Battery', 'Biological', 'Brown-glass', 'Cardboard', 'Clothes', 'Green-glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash', 'White-glass']

# Define a function for predicting with a single image
def predict_single_image(image, model, class_labels):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    return class_labels[predicted_class[0].item()]

# Streamlit UI
st.title('Garbage Classification 🔍')

# Introduction
st.write('**Welcome to the Garbage Classification App!**')
st.write('**This app is to classify images into 12 different classes. The following are the 12 classes that can be predicted:**')

# Explanation of classes in a single row
class_labels_html = ""
for class_label in class_labels:
    class_labels_html += f"<span class='badge badge-secondary'>{class_label}</span>"

# CSS for badge styling
st.markdown("""
    <style>
        .badge {
            display: inline-block;
            font-size: 16px;
            padding: 6px 12px;
            margin: 5px;
            background-color: #007bff;
            color: #fff;
            border-radius: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# Display class labels in a single row
st.markdown(class_labels_html, unsafe_allow_html=True)

# Reduce space between introduction and class labels
st.write("")  # Empty line to reduce space

# Image upload
st.write('Upload an image to predict its class:')
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Predict the class
    image = Image.open(uploaded_image)
    predicted_class = predict_single_image(image, model, class_labels)

    # Display the predicted class
    st.success(f'Predicted Class: **{predicted_class}**')
