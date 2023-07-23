# About 



When compared to image processing and other classification approaches, audio processing is one of the most difficult problems in data science. One such application is music genre classification, which seeks to categorise audio recordings into specific sound genres to which they belong. The application is critical and requires automation to decrease manual error and time since manually classifying music involves listening to each file for the whole duration. So, to automate the process, we employ machine learning and deep learning algorithms, which we will demonstrate in this post.

![OIP (2)](https://github.com/Vipul1019/Music_Genra_Classification/assets/77145832/49190cf4-9a7c-4f09-a749-664695cded89)


In brief, we may express our project issue statement as "given multiple audio files, and the task is to categorise each audio file in a specific category, such as audio belongs to Disco, hip-hop, etc". The top four most commonly used methodologies for music genre classification are shown below:


1. Multiclass support vector machine


2. K-Nearest Neighbors

3. K-means clustering algorithm

4. Convolutional neural network



# Features Of our Model

1. It categorises the input audio sample or song according to the music genre to which it belongs.

2. A model with a higher accuracy rate that can be used in real-time applications.

3. Decreases or eliminates the use of manual activities in music genre classification.

4. Classify the music based on its characteristics.


# Dataset Overview

The GTZAN genre collection dataset, which is a popular audio collecting dataset, will be used. It comprises roughly 1000 audio tracks divided into ten categories. The audio files are all in.wav format (extension). Blues, Hip-hop, classical, pop, Disco, Country, Metal, Jazz, Reggae, and Rock are the genres to which audio files belong. The dataset is easily found on Kaggle and can be downloaded from below given link. 
Link:"https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?select=Data"

We have considered 80% of the dataset as training size and 20% of the dataset as testing size.


# Methodology

1. Read the audio files.

2. Convert the audio files(wav files) to spectrogram in image 
   format.

3. Normalize all the shape of waveforms to one size.

4. Extract features using YAMNet(CNN) architecture.

5. Classify the audio into particular genre.


# Model

We have used "Yamnet" model for our music genra classification model.
Normally CNN are used to classify images but one dimensional filters can be utilized to classify audio. 
Also two dimensional filters can be used in the CNNs with spectrograms of audio.
Several CNNs with general configuration are trained and fine-tuned to classify a set of songs. 
Output of a 10 fully connected layer is saved for each song. This outputs used as feature vector in similarity calculations.


![OIP (3)](https://github.com/Vipul1019/Music_Genra_Classification/assets/77145832/2128c830-4360-4c1f-a6e6-0afdcc3b8b65)


# Why Yamnet?

Resnet18 : it uses the weights of pre-trained ImageNet. it would not allow the vanishing gradient problem to occur. But here the loss and accuracy remains constant after several epochs.


Yamnet : One specific use of YAMNet is as a high-level feature extractor - the 1,024-dimensional embedding output. You will use the base (YAMNet) model's input features and feed them into your shallower model consisting of one hidden (tf.keras.layers.Dense) layer. Then, you will train the network on a small amount of data. Here the loss decreases and accuracy increases as the epochs are increased. 




