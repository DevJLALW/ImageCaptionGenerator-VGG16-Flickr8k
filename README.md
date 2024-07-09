# Image Caption Generator
## Objective
The aim of this project is to generate captions for given input images. The dataset used comprises 8,000 images, each accompanied by five captions. The project leverages both Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs) to extract features from the images and the corresponding text captions.

## Dataset
Source: Flickr8k Dataset on Kaggle:  https://www.kaggle.com/adityajn105/flickr8k

## Steps
1. Environment Setup
Google Collab
2. Feature Extraction
Use the VGG16 model to extract features from images. The VGG16 model is restructured to output features from the last fully connected layer.
Preprocess the images and extract feature vectors for each image in the dataset.
Save the extracted features to a file for later use.
3. Captions Data Preparation
Load the captions from the dataset and map each image to its corresponding captions.
Preprocess the captions by converting them to lowercase, removing special characters and digits, and adding start and end sequence tags.
4. Tokenization
Tokenize the text captions to convert words into integer sequences.
Determine the vocabulary size and the maximum caption length for padding.
5. Data Splitting
Split the dataset into training and testing sets, with 90% of the data used for training and 10% for testing.
6. Data Generator
Implement a data generator to yield image features and corresponding text sequences in batches. This helps in efficiently training the model without loading the entire dataset into memory.
7. Model Architecture
Define the image feature extractor model using CNN.
Define the text processing model using Embedding and LSTM layers.
Combine the outputs of the CNN and LSTM models to form the final image captioning model.
Compile the model using categorical cross-entropy loss and the Adam optimizer.
8. Model Training
Train the model using the data generator for a specified number of epochs.

## Conclusion
This project demonstrates an approach to generate descriptive captions for images using deep learning techniques. By combining CNNs for image feature extraction and LSTMs for sequence processing, the model learns to produce coherent and contextually relevant captions for the input images.
