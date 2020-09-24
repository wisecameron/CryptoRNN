'''
AUTHOR: CAMERON WARNICK
DATE: 09/19/20
THIS SCRIPT WILL CREATE A RECURRENT NEURAL NETWORK THAT CAN BE APPLIED TO ANY CRYPTOCURRENCY.
THIS SCRIPT IS EXTREMELY BASIC AND DOES NOT PROVIDE TECHNICAL INDICATORS, SENTIMENT, ETC.
LONG STORY SHORT, PROBABLY IS NOT PROFITABLE.  PLEASE USE AS A BASELINE AND DO NOT
TRADE SOLELY BASED OFF OF THE PREDICTIONS MADE BY THIS ALGORITHM.

INPUT: DATASET CONTAINING PRICE HISTORY IS REQUIRED, MODEL CAN PROCESS NUMERICAL DATA
AND WILL SCALE IT TO PCT_CHANGE.
OUTPUT: 0/1 (PRICE DECREASE/PRICE INCREASE) IN EXPRESSED TIME SLOT (FUTURE_PERIOD_PREDICT)

General Thoughts/ Intuition
What are we actually doing?
The goal of this mdoel is not only to determine whether the price will increase or decrease,
but also to test which features are relevant to the price action, which can be useful
in the future for more advanced models or for your own technical analysis.

Using Data:
It is important to have an even split between 0 and 1 in the dependent variable vector,
as a bias towards buy or sell would be translated into the model.
The data is also shuffled multiple times to create a dynamic and truly random orientation,
which helps the model to find genuine connections instead of overfitting or cheating.
Furthermore, pct_change is implemented onto volume and price history to help the model
understand the data (unscaled data can lead to issues with some types of neural networks)

Please Note: I based this model off of a tutorial I followed found here:
    https://pythonprogramming.net/crypto-rnn-model-deep-learning-python-tensorflow-keras/

The main point of this project is to update the model above to work in 2020, 
as many aspects of the code are outdated and it will most likely not debug
on the newest versions of tensorflow, keras, and python itself.
There were many new debugging issues that have now been solved, and all of the
original extensions from the original project are still intact & functioning
 (ie: checkpoints, tensorboard)

My goal from this project is to improve my understanding of these 
concepts and can implement them successfully, which is why I have added several independent features
to this model and also used a separate dataset.  I also wrote notes
without referring to the tutorial to make sure I really understand the greater concepts.
Still, it should be noted that I used these tutorials to learn how to make this model.
I intend to continue learning and do not claim that I am currently this skilled,
I am entirely focused on building as deep of an understanding as I can.

I am currently working on my own crypto buy/sell algorithm which implements several
technical indicators. They are used to evaluates the price history to raise buy or sell signals (0/1)
I mostly did this project to understand how to correctly process data, 
improve my understanding of deep learning models, and figure out how to 
effectively implement an algorithm based on a tradeable asset.  Please note
that I in no way take credit for the code or intuition below as it is largely the same
as what is shown in the tutorial. 
'''



#imports
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, LSTM, BatchNormalization)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
#ModelCheckpoint always saves your best model (lowest error!)
#So you could run with 100 epochs and know your model will eventually overfit,
#but have the most suitable amount of epochs displayed after training.

#Global variables
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 10
BATCH_SIZE = 64

#Create unique name for each model parameter to represent them accurately and distinctly
NAME = f'{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'

#----------------------------------------------------------------------------#
def classify(current, future):
    '''
    this function will compare future and current price values to determine
    whether the price increased or decreased over the given interval.
    IN: current price, future price
    OUT: int(0/1) price increase (t/f)
    '''
    
    if float(future) > float(current):
        return 1
    else:
        return 0
#----------------------------------------------------------------------------#
def preprocess_dataset(dataset):
    
    '''
    this function will preprocess the data, creating a matrix of features (x)
    and a dependent variable vector (y).  X will contain all columns barring 'target',
    scaled to represent pct_change.  The target (price increase y/n is represented with y).
    NOTE: If running a unique dataset, bar any values that do not require feature scaling
    and append them as a np array to the matrix of features after data preprocessing.
    NOTE PART 2: SEQ_LEN determines the length of the sequence.  Will only return
    that many values.
    
    IN: Dataset 
    OUT: matrix of features, dependent variable vector.
    '''
    
    #Would obviously ruin the model to include the future price in the dataset.
    dataset = dataset.drop('future', 1) 
    
    #This loop is what normalizes the data with pct_change.  It goes through
    #each column, not each value.
    for col in dataset.columns:
        if col != 'target':
            #Normalize with pct change
            dataset[col] = dataset[col].pct_change()
            #drop NaN
            dataset.dropna(inplace = True)
            dataset[col] = preprocessing.scale(dataset[col].values)
    
    #NOTE: dropna removes any NaN values.
    #dropna again for any created by pct_change
    dataset.dropna(inplace = True)
    
    #Sequential data list
    sequential_data = []
    
    #Create deque - when deque hits len = maxlen, will automatically pop old values to add new.
    prev_days = deque(maxlen = SEQ_LEN)
    
    for i in dataset.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            #append x & y - features & label
            sequential_data.append([np.array(prev_days), i[-1]])
        
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
          
    #Identify which list has fewer values, make lower == len(smallest)
    lower = min(len(buys), len(sells))
        
    #Equalize Buys and Sells to avoid creating too much bias in the model.
    #Note: target is not computed with any advanced math, it is simply telling whether
    #The future price will be higher than the current price.  Based on the logic
    # in the function classify.
    buys = buys[:lower]
    sells = sells[:lower]
        
    #Shuffle again so the data isn't split 50/50 between buys and sells
    #ie: buy, buy.... sell, sell, sell, sell, sell, etc
    random.shuffle(sequential_data)
        
    #X contains the matrix of features, y contains the dependent variable vector.
        
    x = []
    y = []
        
    #Sequence contains matrix of features, scaled to pct change, target contains
    #Is target (0/1)
    for sequence, target in sequential_data:
        x.append(sequence)
        y.append(target)
            
            
    return np.array(x), y
#----------------------------------------------------------------------------#
'''Pre-training'''
    
#Initialize main dataset (or dataframe) - will contain final values (prior to preprocessing)
main_dataset = pd.DataFrame()
#Keep track of ratio name
ratioName = ['KMD_USD']

for ratio in ratioName:
    datalink = f'{ratio}.csv'
    dataset = pd.read_csv('price_history.csv')
    dataset.set_index("date", inplace = True) #inplace = edit original object
    
    if(len(main_dataset) == 0):
        main_dataset = dataset
    else:
        main_dataset = main_dataset.join(dataset)
    
main_dataset['future'] = main_dataset['close'].shift(+FUTURE_PERIOD_PREDICT)
main_dataset['target'] = list(map(classify, main_dataset['close'], main_dataset['future']))

times = sorted(main_dataset.index.values)
last_5pct = times[-int(0.05*len(times))]

#Separate validation and training data
validation_main_dataset = main_dataset[(main_dataset.index >= last_5pct)]
main_dataset = pd.DataFrame(data = main_dataset[main_dataset.index < last_5pct])


x_train, y_train = preprocess_dataset(main_dataset)
x_validate, y_validate = preprocess_dataset(validation_main_dataset)

#----------------------------------------------------------------------------#
'''Model training'''

model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))
opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)

#COMPILE MODEL
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt,
              metrics = ['accuracy'])


tensorboard = TensorBoard(log_dir = f'logs/{NAME}')
#Checkpoint object - copy and pasted
filepath = "RNN_Final-{epoch:02d}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy',
                        verbose=1, save_best_only=True, mode='max')) # saves only the best one


#Avoiding ValueError raised if training / validation sets contain lists instead of np array
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_validate = np.asarray(x_validate)
y_validate = np.asarray(y_validate)



history = model.fit(x_train, y_train,
                    batch_size = BATCH_SIZE,
                        epochs = EPOCHS,
                        validation_data= (x_validate, y_validate),
                        callbacks=[tensorboard, checkpoint])
