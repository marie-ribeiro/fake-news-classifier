#!/usr/bin/python
# LSTM for sequence classification for HOMUS dataset
import numpy
import tensorflow as tf
from keras.layers import Activation
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers
from numpy import *
from tensorflow.keras import layers
from keras.layers import TimeDistributed
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from tensorflow.keras.utils import to_categorical

import socket
import numpy
import tensorflow as tf
from keras.layers import Activation
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers
from numpy import *
from tensorflow.keras import layers
from keras.layers import TimeDistributed
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
#Version should be 2.0.0
print(tf.version.VERSION)
#seed(32)
#Preprocessing Data
#Parameters
#Limit on the number of features. Using the top 20,000 features
TOP_K = 20000
#Limit on the length of text sequences, longer sequences will be truncated
MAX_SEQUENCE_LENGTH = 500

# Load LSTM
json_file = open('classifierFakeRealNews.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("classifierFakeRealNews.h5")
print("Loaded model from disk")

#Process Data
csvFake = pd.read_csv('Fake.csv')
csvTrue = pd.read_csv('True.csv')
textFake_df = pd.DataFrame(csvFake[['title','text']])
textFake_df['label'] = 0  #adding label column to dataframe
textTrue_df = pd.DataFrame(csvTrue[['title','text']])
textTrue_df['label'] = 1  #adding labels

frames = [textTrue_df, textFake_df]
dataset = pd.concat(frames) #merging real and fake dataset
#This Shuffles the dataset and splits into 80% for training and 20% for testing 
#as a result, train set = 35,918 and test set = 8,980
trainTexts, testTexts = train_test_split(dataset, test_size=0.2)

def tokenize_vectorize(trainTexts, testTexts):
    #Tokenization and Vectorisation for sequence models, this method assumes that order of words is important in text, and is better for CNN and RNN
    #Create vocabulary with training texts
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(trainTexts.text)
    word_index_text = tokenizer.word_index
    #Create vocabulary with training title
    tokenizer.fit_on_texts(trainTexts.title)
    word_index_title = tokenizer.word_index    
    #Vectorize the training and validation texts
    trainSetText = tokenizer.texts_to_sequences(trainTexts.text)
    testSetText = tokenizer.texts_to_sequences(testTexts.text)
    trainSetTitle = tokenizer.texts_to_sequences(trainTexts.title)
    testSetTitle = tokenizer.texts_to_sequences(testTexts.title)    
    
    #Get max sequence length
    max_length = len(max(trainSetText, key =len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH
        
    #Fix sequence length to max value. 
    #The sequence is padded in the beginning if shorter than the length
    #and longer sequences are truncated
    trainSetText = sequence.pad_sequences(trainSetText, maxlen = max_length)
    trainSetTitle = sequence.pad_sequences(trainSetTitle, maxlen = max_length)
    testSetText = sequence.pad_sequences(testSetText, maxlen = max_length)
    testSetTitle = sequence.pad_sequences(testSetTitle, maxlen = max_length)
    trainSetText = numpy.array(trainSetText)
    trainSetTitle = numpy.array(trainSetTitle)
    testSetText = numpy.array(testSetText)
    testSetTitle = numpy.array(testSetTitle)
    
    trainSetText =  trainSetText.reshape((trainSetText.shape[0], trainSetText.shape[1], 1))
    testSetText =  testSetText.reshape((testSetText.shape[0], testSetText.shape[1], 1))
    trainSetTitle = trainSetTitle.reshape((trainSetTitle.shape[0], trainSetTitle.shape[1], 1))
    testSetTitle =  testSetTitle.reshape((testSetTitle.shape[0], testSetTitle.shape[1], 1))
    #Shape should be 35918, 500, 1

    X_train = [trainSetText, trainSetTitle]
    X_test = [testSetText, testSetTitle]

    #Labels- Converting labels to binary vectors  
    Y_train = to_categorical(trainTexts.label, num_classes=2)
    Y_test = to_categorical(testTexts.label, num_classes=2)

    return X_train, X_test, Y_train, Y_test, word_index_text, word_index_title

X_train, X_test, Y_train, Y_test, word_index_text, word_index_title = tokenize_vectorize(trainTexts, testTexts)

#Send through Classifier

#prediction = loaded_model.predict_classes(X_train)
#predicted = classOptions[prediction[0]]
#print("Predicted=%s" % (predicted[0]))

predicted = loaded_model.predict(X_train)
print("Predicted=%s" % (predicted[1]))
