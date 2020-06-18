import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from tensorflow.keras.utils import to_categorical

#Parameters
#Limit on the number of features. Using the top 20,000 features
TOP_K = 20000
#Limit on the length of text sequences, longer sequences will be truncated
MAX_SEQUENCE_LENGTH = 500

#Data stats:
#Number of true entries; 21,417
#Number of false entries; 23,481
#Total entries; 44,898

csvFake = pd.read_csv('Fake.csv') 
csvTrue = pd.read_csv('True.csv')
textFake_df = pd.DataFrame(csvFake['text'])
textFake_df['label'] = 0  #adding label column to dataframe
textTrue_df = pd.DataFrame(csvTrue['text'])
textTrue_df['label'] = 1  #adding labels

frames = [textTrue_df, textFake_df]
dataset = pd.concat(frames) #merging real and fake dataset

#This Shuffles the dataset and splits into 80% for training and 20% for testing 
#as a result, train set = 35,918 and test set = 8,980
trainTexts, testTexts = train_test_split(dataset, test_size=0.2)
#print(trainTexts.head())

def tokenize_vectorize(trainTexts, testTexts):
    #Tokenization and Vectorisation for sequence models, this method assumes that order of words is important in text, and is better for CNN and RNN
    #Create vocabulary with training texts
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(trainTexts.text)
    word_index = tokenizer.word_index
    #Vectorize the training and validation texts
    trainSet = tokenizer.texts_to_sequences(trainTexts.text)
    testSet = tokenizer.texts_to_sequences(testTexts.text)
    #Get max sequence length
    max_length = len(max(trainSet, key =len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    #Fix sequence length to max value. 
    #The sequence is padded in the beginning if shorter than the length
    #and longer sequences are truncated
    X_train = sequence.pad_sequences(trainSet, maxlen = max_length)
    X_test = sequence.pad_sequences(testSet, maxlen = max_length)
    #print(trainSet)

    #Labels- Converting labels to binary vectors  
    Y_train = to_categorical(trainTexts.label, num_classes=2)
    Y_test = to_categorical(testTexts.label, num_classes=2)

    return X_train, X_test, Y_train, Y_test, word_index

X_train, X_test, Y_train, Y_test, word_index = tokenize_vectorize(trainTexts, testTexts)
#print(X_train)