#got the requirements with this command:     pip freeze > requirements.txt


#     conda create -n envConda_docker python=3.11
#     conda activate envConda_docker
#     pip install -r requirements.txt

import torch
from torchvision import transforms, models, datasets
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time


class ImagePredictor:
    def __init__(self, model_path):
        # Initialize the model structure
        self.model = models.resnet18(pretrained=False)
        
        # Adjust the output size of the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)
        
        # Load the state dictionary
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.25, 0.25, 0.25])
        
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch
    
    def predict_image_class(self, image_path):
        input_batch = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(input_batch)
        _, predicted = torch.max(output.data, 1)
        return predicted
    
    def get_class_names(self, data_dir):        
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
        }
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}
        
        class_names = image_datasets['train'].classes
        return class_names
    
    def predict_class_name(self, image_path):
        predicted_class = self.predict_image_class(image_path)
        
        data_dir = '/home/prgrmcode/app/data/data_beexant/archive/hymenoptera_data/'
        # data_dir = '../data/data_beexant/archive/hymenoptera_data/'
        class_names = self.get_class_names(data_dir)
        predicted_class_name = class_names[predicted_class[0]]
        
        print(f"Predicted class for image {image_path}: {predicted_class_name}")
        image = Image.open(image_path)
        
        plt.imshow(image)
        plt.title(f'Predicted class: {predicted_class_name}')
        plt.show()
        return predicted_class_name

# Load the trained model
# model_path = '/home/prgrmcode/app/model/model.pth'
# while not os.path.exists(model_path):
#     print('waiting for model to be created by train script...')
#     time.sleep(10) # wait 10 seconds

# Create an instance of the predictor class
# predictor = ImagePredictor(model_path)
# predictor.show_image('predict1.webp')
# predictor.show_image('predict2.webp')
# predictor.show_image('predict3.jpg')
# predictor.show_image('predict4.webp')
# predictor.show_image('predict5.jpg')
# predictor.show_image('predict6.jpg')




# to make it without a class:

# # Load the trained model
# model = torch.load('../train/model.pth')
# model.eval()

# # Define the same transforms used during training
# mean = np.array([0.5, 0.5, 0.5])
# std = np.array([0.25, 0.25, 0.25])


# # Preprocess input image
# def preprocess_image(image_path):
#     image = Image.open(image_path)
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#     input_tensor = transform(image)
#     input_batch = input_tensor.unsqueeze(0)
#     return input_batch

# # Perform inference
# def predict_image_class(image_path):
#     input_batch = preprocess_image(image_path)
#     with torch.no_grad():
#         output = model(input_batch)
#     # Process the output as needed
#     _, predicted = torch.max(output.data, 1)
#     return predicted


# # Usage
# image_path = 'predict1.webp'
# predicted_class = predict_image_class(image_path)
# print(f"Predicted class for first image: {predicted_class}")
# # show the first image with its predicted class. make a class for it:

# image = Image.open(image_path)
# plt.imshow(image)
# plt.title(predicted_class[0])
# plt.show()
# print(type(predicted_class))
# print(predicted_class)


# image_path = 'predict2.avif'
# predicted_class = predict_image_class(image_path)
# print(f"Predicted class for second image: {predicted_class}")

# image_path = 'predict3.jpg'
# predicted_class = predict_image_class(image_path)
# print(f"Predicted class for third image: {predicted_class}")
