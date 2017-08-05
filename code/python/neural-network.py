# "pip install keras scikit-learn" if you have not already done so
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import boston_housing
from sklearn.metrics import mean_absolute_error

# load boston housing data. Uses 80%/20% train/test split by default.
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# create our neural network. Will have two hidden layers of 64 units each
# and an output layer with one neuron because we are predicting a single 
# scalar value. Use relu activation functions.
model = Sequential()
# add a fully-connected layer with 64 units
model.add(Dense(64, input_shape=(13,)))
# apply a relu activation function to "bend" the linear output of
# the fully connected layer
model.add(Activation('relu'))
# repeat...
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))

# use RMSProp for optimization and Mean Squared Error (MSE) as loss function.
model.compile(optimizer='rmsprop', loss='mse')
model.summary()

print('[*] training model...')
model.fit(x_train, y_train, epochs=150, verbose=2)

print('[*] predicting from test set...')
y_hat = model.predict(x_test)

for i in range(len(y_hat)):
	print ('[+] predicted: {:.1f}    real: {}     error: {:.1f}'\
		   .format(y_hat[i][0], y_test[i], abs(y_hat[i][0] - y_test[i])))

print('[+] the mean absolute error is {:.1f}'.format(mean_absolute_error(y_hat, y_test)))