# LSTM for sequence classification for HOMUS dataset
from numpy.random import seed #trying to have consistent results. both are apparently needed
seed(1)
import tensorflow as tf
import keras as keras
tf.random.set_seed(1)
from keras.layers import Activation
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.layers import Dense, Conv2D, LSTM, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Input
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
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import tokenize_vectorize

#Version should be 2.0.0
print(tf.version.VERSION)

#Data stats:
#Number of true entries; 21,417
#Number of false entries; 23,481
#Total entries; 44,898
MAX_SEQUENCE_LENGTH = 500

csvFake = pd.read_csv('Fake.csv')
csvTrue = pd.read_csv('True.csv')
textFake_df = pd.DataFrame(csvFake[['title','text']])
textFake_df['label'] = 0  #adding label column to dataframe
textTrue_df = pd.DataFrame(csvTrue[['title','text']])
textTrue_df['label'] = 1  #adding labels

frames = [textTrue_df, textFake_df]
dataset = pd.concat(frames) #merging real and fake dataset
#This Shuffles the dataset and splits into 80% for training and 20% for testing 
#as a result, train set = 35,918 and test set = 8,980. Random_state sets seed for splitting 
trainTexts, testTexts = train_test_split(dataset, test_size=0.2, random_state=1)
X_train, X_test, Y_train, Y_test, word_index_text, word_index_title = tokenize_vectorize(trainTexts, testTexts)
text_vocab_size = len(word_index_text) + 1
title_vocab_size = len(word_index_title) + 1

#Network
def networkForArticalText():
    global MAX_SEQUENCE_LENGTH
    #Takes genre of article and adds to first model
    model1_in = Input(shape=(MAX_SEQUENCE_LENGTH,))
    model1_layer1 = Embedding(input_dim=text_vocab_size, output_dim=100, input_length=(MAX_SEQUENCE_LENGTH,))(model1_in)#Dense(40, activation='relu')(model1_in)
    model1_layer2 = Flatten()(model1_layer1)
    model1_layer3 = Dense(2, activation='relu')(model1_layer2)
    model1_layer4 = Dense(2, activation='relu')(model1_layer3)
    model1_out = Dense(1, activation='relu')(model1_layer4)
    model1 = Model(model1_in, model1_out)
    return(model1)

def networkForTitle():
    global MAX_SEQUENCE_LENGTH
    model2_in = Input(shape=(MAX_SEQUENCE_LENGTH,))
    model2_layer1 = Embedding(input_dim=title_vocab_size, output_dim=100, input_length=(MAX_SEQUENCE_LENGTH,))(model2_in)
    model2_layer2 = LSTM(2, activation='relu',
                      return_sequences=True, dropout=0.1, stateful=False, name='layer2Title')(model2_layer1)
    model2_out = Flatten()(model2_layer2)
    model2 = Model(model2_in, model2_out)
    return(model2) 

 '''
    #alternative faster version but lacks lstm--to del
    global MAX_SEQUENCE_LENGTH
    #Takes titles and adds to second model
    model2_in = Input(shape=(MAX_SEQUENCE_LENGTH,))
    model2_layer1 = Embedding(input_dim=title_vocab_size, output_dim=100, input_length=(MAX_SEQUENCE_LENGTH,))(model2_in)
    model2_layer2 = Flatten()(model2_layer1)
    model2_out = Dropout(0.2)(model2_layer2)
    model2 = Model(model2_in, model2_out) 
    return(model2)'''

def networksCombined(model1, model2):
    #Combines 2 models and gives output
    concatenated = concatenate([model1.output, model2.output])
    out = Dense(2, activation='softmax', name='output')(concatenated)
    model = Model([model1.input, model2.input], outputs = out)
    return(model)
    
model1 = networkForArticalText()
model2 = networkForTitle()
model = networksCombined(model1, model2)
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),#
              #optimizer='Adam', #optimizer only changed to solve userwarning message about 'sparse indexed slices' that occurs with using the embedding layer
              metrics=['accuracy']) 
model.fit(
    X_train,
    Y_train,
    epochs = 10, validation_split=0.1,
    batch_size = 100,
    verbose = 2 #Removes warnings
)

#Testing
scores = model.evaluate(X_test, Y_test, batch_size = 100, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100), flush=True)
    
model_json = model.to_json()
with open("classifierFakeRealNews.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("classifierFakeRealNews.h5")
print("Saved model to disk", flush=True)
#Once complete: make java app for use
