# pip install scikit-learn if you have not already done so
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# load boston housing data
data = datasets.load_boston()

# use %80 percent of the data for training and 20% for testing
y_train = data.target[0:int(len(data.target) * 0.80)]
y_test  = data.target[int(len(data.target) * 0.80):]
X_train = data.data[0:int(len(data.data) * 0.80)]
X_test  = data.data[int(len(data.data) * 0.80):]

model = LinearRegression(normalize=True)

print('[*] training model...')
model.fit(X_train, y_train)

print('[*] predicting from test set...')
y_hat = model.predict(X_test)

# print the results
for i in range(len(y_hat)):
	print ('[+] predicted: {:.1f}    real: {}     error: {:.1f}'\
		   .format(y_hat[i], y_test[i], abs(y_hat[i] - y_test[i])))

print('[+] the mean absolute error is {:.1f}'.format(mean_absolute_error(y_hat, y_test)))