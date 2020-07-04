import pandas as pd 
import numpy as numpy
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from tensorflow.keras.utils import to_categorical

#Preprocessing Data
#Parameters
#Limit on the number of features. Using the top 20,000 features
TOP_K = 20000
#Limit on the length of text sequences, longer sequences will be truncated
MAX_SEQUENCE_LENGTH = 500

def tokenize_vectorize(trainTexts, testTexts):
    #Tokenization and Vectorisation for sequence models, this method assumes that order of words is important in text, and is better for CNN and RNN
    #Create vocabulary with training texts
    tokenizer = text.Tokenizer(num_words=TOP_K, lower=False)
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

    #Commented out to fit with other changes to the classifier model
    #trainSetText =  trainSetText.reshape((trainSetText.shape[0], trainSetText.shape[1], 1))
    #testSetText =  testSetText.reshape((testSetText.shape[0], testSetText.shape[1], 1))
    #trainSetTitle = trainSetTitle.reshape((trainSetTitle.shape[0], trainSetTitle.shape[1], 1))
    #testSetTitle =  testSetTitle.reshape((testSetTitle.shape[0], testSetTitle.shape[1], 1))
    #Shape should be 35918, 500, 1

    X_train = [trainSetText, trainSetTitle]
    X_test = [testSetText, testSetTitle]

    #Labels- Converting labels to binary vectors  
    Y_train = to_categorical(trainTexts.label, num_classes=2)
    Y_test = to_categorical(testTexts.label, num_classes=2)

    return X_train, X_test, Y_train, Y_test, word_index_text, word_index_title