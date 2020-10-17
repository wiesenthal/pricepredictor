from statistics import median, variance
from argparse import ArgumentParser
#argument parsing
parser = ArgumentParser()
parser.add_argument('--batch_size', help='Size of each batch.', type=int, default=32)
parser.add_argument('--num_trains', help='Number of times to train the model.', type = int, default = 1)
parser.add_argument('--num_epochs', help='Number of epochs per train.', type = int, default = 230)
parser.add_argument('--testing_mode', help='For using testing and training data, or just training.', dest='testing_mode', default=True, action='store_true')
parser.add_argument('--tbm', help='Test batch multiplier, test batch = tbm*batch_size', type = int, default = 7)
parser.add_argument('--plot_mode', help='When plotting the data on a graph.', dest='plot_mode', default=True, action='store_true')
args = parser.parse_args()

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras import optimizers
from tensorflow import compat
import matplotlib.pyplot as plt

import data_parse
from tensorflow.python.distribute.device_util import current

np.set_printoptions(suppress=True)
np.set_printoptions( linewidth=100)
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

#Data parameters
df, columns = data_parse.parse_data()
df, amax = data_parse.scale_data(df)
num_points = data_parse.num_points(df)
#split into samples to be trained on for LSTM
def split_sequences(data, steps):
    X, y = list(), list()
    for i in range(data.shape[0]):
        end = i + steps #get end of sample
        if end > data.shape[0] - 1:
            #if past the dataset
            break
        #input and outputs
        seq_x = data[i:end,:]
        seq_y = data[end,4]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def hist_plot(history, num, fig, axs):
    #plot the history
    training_loss = history.history['loss']
    if testing_mode:
        testing_loss = history.history['val_loss']
    epoch_count = range(1, len(training_loss) + 1)
    if num_trains == 1:
        axs.plot(epoch_count, training_loss, 'r--')
        if testing_mode:
            axs.plot(epoch_count, testing_loss, 'b-') #when using testing data
            axs.legend(['Training Loss', 'Testing Loss']) #when using testing data
        else:
            axs.legend(['Loss']) #when using training data only
        axs.set_title('Model #' + str(num + 1))
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Loss')
    else:
        axs[num].plot(epoch_count, training_loss, 'r--')
        if testing_mode:
            axs[num].plot(epoch_count, testing_loss, 'b-') #when using testing data
            axs[num].legend(['Training Loss', 'Testing Loss']) #when using testing data
        else:
            axs[num].legend(['Loss']) #when using training data only
        axs[num].set_title('Model #' + str(num + 1))
        axs[num].set_xlabel('Epoch')
        axs[num].set_ylabel('Loss')
######################
#number of time_steps per batch (batch size)
n_steps = args.batch_size
#number of times to train the network from anew (in order to take multiple predictions)
num_trains = args.num_trains
#number of epochs per train
num_epochs = args.num_epochs
######################

testing_mode = args.testing_mode #true if using train and test data, false if not
plot_mode = args.plot_mode #true if plotting data
tbm = args.tbm #test batch multiplier
#args from ArgumentParser

if not testing_mode:
    X, y = split_sequences(df,n_steps) #when using just training
else:
    X, y = split_sequences(df[0:-1*n_steps*tbm,:], n_steps)
    test_d = df[-1*n_steps*tbm:,:]
    test_X, test_y = split_sequences(test_d, n_steps)

n_features = X.shape[2]

if plot_mode:
    fig, axs = plt.subplots(num_trains) #for plotting the data

def getpercent(right, wrong):
    return round(float(100*right/(right + wrong)), 2)

def train_network(epochs, do_print=True):
    model = Sequential()
    #CuDNNLSTM for GPU support
    model.add(LSTM(100, return_sequences=True, input_shape=(n_steps,n_features)))
    model.add(Dense(1))
    model.compile(optimizer='Adagrad', loss='mse')

    if testing_mode:
        history = model.fit(X, y, epochs=epochs, batch_size=n_steps, validation_data=(test_X, test_y), verbose=2) #when using training/testing data
    else:
        history = model.fit(X, y, epochs=230, batch_size=n_steps, verbose=2) #when using just training data

    if plot_mode:
        hist_plot(history, i, fig, axs) #For plotting the data
    ###################################################
    #predict
    recent = df[-1*n_steps:,:]
    recent = recent.reshape((1, n_steps, n_features))
    result = model.predict(recent, batch_size = n_steps, verbose=0)
    result = (result*amax[4])[0][0]

    #get errors
    c, f, tr, tf, zr, zf, xr, xf, hr, hf = error(model)
    print(f'correct predictions {getpercent(c, f)}%')
    print(f'correct +10 {getpercent(tr, tf)}%')
    print(f'correct +20 {getpercent(zr, zf)}%')
    print(f'correct +40 {getpercent(xr, xf)}%')
    print(f'correct +80 {getpercent(hr, hf)}%')

    return(result)

def error(model, num=600):
    l = df.shape[0]
    startIndex = l - num - n_steps
    correct = 0
    tenRight = 0
    tenWrong = 0
    twenRight = 0
    twenWrong = 0
    right40 = 0
    wrong40 = 0
    right80 = 0
    wrong80 = 0
    false = 0
    for i in range(num - 1):
        start = startIndex + i
        recent = df[start:start+n_steps,:]
        recent = recent.reshape((1, n_steps, n_features))
        result = model.predict(recent, batch_size = n_steps, verbose = 0)
        prediction = (result*amax[4])[0][0]
        on_day = df[start + n_steps - 1,4] * amax[4]
        next_day = df[start + n_steps, 4] *amax[4]
        if ((next_day > on_day and prediction > on_day) or (next_day < on_day and prediction < on_day)):
            correct += 1
            if abs(prediction - on_day) > 10:
                tenRight += 1
            if abs(prediction - on_day) > 20:
                twenRight += 1
            if abs(prediction - on_day) > 40:
                right40 += 1
            if abs(prediction - on_day) > 80:
                right80 += 1
        else:
            false += 1
            if abs(prediction - on_day) > 10:
                tenWrong += 1
            if abs(prediction -on_day) > 20:
                twenWrong += 1
            if abs(prediction - on_day) > 40:
                wrong40 += 1
            if abs(prediction - on_day) > 80:
                wrong80 += 1
    return correct, false, tenRight, tenWrong, twenRight, twenWrong, right40, wrong40, right80, wrong80


prices = []
for i in range(num_trains):
    print('')
    print('Iteration ' + str(i + 1) + '/' + str(num_trains))
    p = train_network(num_epochs)
    print('Predicted price of Ethereum: $' + str(p))
    prices.append(p)
#displaying results
current_price = (df[-1,4]*amax[4])

if (num_trains > 1):
    mean = sum(prices)/num_trains
    med = median(prices)
    ran = max(prices)-min(prices)
    var = variance(prices, mean)
    print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('All ' + str(num_trains) + ' price predictions (sorted).')
    print(sorted(prices))
    print('Variance: ' + str(round(var, 3)) + '    Range: ' + str(round(ran, 3)))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Price of Ethereum today: $' + str(current_price))
    print('Mean predicted price of Ethereum tomorrow: $' + str(mean))
    print('Median predicted price of Ethereum tomorrow: $' + str(med))
else:
    print('Price of Ethereum today: $' + str(current_price))
if plot_mode:
    plt.show()

