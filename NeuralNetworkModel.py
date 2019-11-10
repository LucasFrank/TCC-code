import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GRU
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import EarlyStopping

case = 2

def custom_function(alpha):
	if case == 1:
		def custom_activation(x):
			return 1 / (1 + K.exp(-alpha * x))
		return custom_activation
	elif case == 2:
		def custom_activation(x):
			return K.relu(x)
		return custom_activation
	elif case == 3:
		def custom_activation(x):
			return K.softplus(x)
		return custom_activation

class ModelMLP:
	
	def __init__(self, n, shape, epochs, learning_rate, alpha):
		cf = 'custom_activation'
		self.model = Sequential()

		get_custom_objects().update({cf: Activation(custom_function(alpha))})
		#self.model.add(GRU(n, input_shape = shape, return_sequences = True, activation = cf, kernel_initializer = 'normal'))
		self.model.add(Dense(units = n,input_shape = shape ,activation = cf, kernel_initializer='normal'))
		#self.model.add(GRU(n, return_sequences = False, activation = cf, kernel_initializer = 'normal'))
		#self.model.add(Dense(units = n, activation = cf, kernel_initializer = 'normal'))
		self.model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'normal'))
		decay_rate = learning_rate/epochs;
		adam = Adam(lr = learning_rate,decay = decay_rate)
		#sgd = SGD(lr = learning_rate, decay = decay_rate)
		self.model.compile(loss = 'mean_squared_error', optimizer = adam)
	
	def fit(self, X1_train, Y1_train, epochs, verbose, callbacks):
		return self.model.fit(X1_train, Y1_train, epochs = epochs, verbose = 0, callbacks = callbacks)
		 
	def predict(self, X1_test):
		return self.model.predict(X1_test)
		
	def clearModel(self):
		K.clear_session()
		
class ModelGRU:
	
	def __init__(self, n, shape, epochs, learning_rate, alpha):
		cf = 'custom_activation'
		self.model = Sequential()

		get_custom_objects().update({cf: Activation(custom_function(alpha))})
		self.model.add(GRU(n, input_shape = shape, return_sequences = True, activation = cf, kernel_initializer = 'normal'))
		#self.model.add(Dense(units = n,input_shape = shape ,activation = cf, kernel_initializer='normal'))
		self.model.add(GRU(n, return_sequences = False, activation = cf, kernel_initializer = 'normal'))
		#self.model.add(Dense(units = n, activation = cf, kernel_initializer = 'normal'))
		self.model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'normal'))
		decay_rate = learning_rate/epochs;
		adam = Adam(lr = learning_rate,decay = decay_rate)
		#sgd = SGD(lr = learning_rate, decay = decay_rate)
		self.model.compile(loss = 'mean_squared_error', optimizer = adam)
	
	def fit(self, X1_train, Y1_train, epochs, verbose, callbacks):
		return self.model.fit(X1_train, Y1_train, epochs = epochs, verbose = 0, callbacks = callbacks)
		 
	def predict(self, X1_test):
		return self.model.predict(X1_test)
		
	def clearModel(self):
		K.clear_session()