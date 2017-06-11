import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    #Parameter to save the current value of the series 
    index = 0
    seriesLength = len(series)
    while((index + 1 + window_size) <= seriesLength):
        #Array containing the input of the input/output pair
        input_array = []
        #Add the input values for the given window size
        for i in range (index, index + window_size):
            input_array.append(series[i])
        #Add the output of the input/output pair
        y.append(series[index + window_size])
        #Add the input of the input/output pair
        X.append(input_array)
        index += 1

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    X,y = window_transform_series(series = dataset,window_size = window_size)
    # split our dataset into training / testing sets
    train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point

    # partition the training set
    X_train = X[:train_test_split,:]
    y_train = y[:train_test_split]

    # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
    X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
    # Initialize the the Sequential model. It is a linear stack of layers.
    model = Sequential()
    # Add a recurrent layer LSTM. First parameter is the number of hidden units. Second parameter is for the input shape (batch_size, output_dim)
    model.add(LSTM(5, input_shape = (7,1)))
    # Add a fully connected layer of 1 unit. Parameter is the number of units
    model.add(Dense(1))
    model.summary()

    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # run your model!
    model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=0)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    test_text = text
    valid_english_characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '!', ',', '.', ':', ';', '?']
    for i in range(0,len(valid_english_characters)):
        #Leave in the test_text only characters that are not valid english characters.
        test_text = test_text.replace(valid_english_characters[i],'')
    #List all non-valid english characters.
    nonvalid_english_characters = list(set(test_text))
    #Print all non-valid english characters to check if there is any character that shouldn't be a  non-valid english character
    print(nonvalid_english_characters)
    #Based on the characters analysis the following define the final non-valid english characters
    nonvalid_english_characters = ['/', '(', '"', '&', '%', '*', ')', '-', '$', '@', "'"]
    #Remove non-valid english characters from text 
    for i in range(0,len(nonvalid_english_characters)):
        text = text.replace(nonvalid_english_characters[i],'')

    
    # shorten any extra dead space created above
    text = text.replace('  ',' ') 


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # Last element of the input/output pair
    lastElement = window_size
    # Loop until the lastElement of the input/output pair is the last element of the text
    while(lastElement<len(text)):
        inputs.append(text[lastElement-window_size:lastElement])
        outputs.append(text[lastElement])
        lastElement += step_size

    
    return inputs,outputs
