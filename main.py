import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import numpy as np
import random
import pandas as pd
import csv
import time
import math
import random
from keras.callbacks import EarlyStopping
from PSOPopulation import PSO
from NeuralNetworkModel import ModelMLP, ModelGRU

def MAPE(y_true, y_pred):
	errors = 0
	for i in range(len(y_true)):
		errors = errors + (abs((float(y_true[i]) - float(y_pred[i]))) / float(y_true[i]))
	return errors / len(y_pred)


# Variaveis Fixadas
MAX_Q = 3
p = 11 # p = 10
q = 3 # q = 4  & n = 90 it scored a 5.92\% average MAPE and its best performance resulted in a MAPE of 5.37
epochsN = 100 # modificar talvez

# Variaveis a ser otimizadas
n = 100
learning_rate = 0.0005
alpha = 1

# Variaveis PSO
num_of_iterations = 20
population_size = 20
testNumber = 5

n_MAX = 140
n_MIN = 60
alpha_MAX = 52
alpha_MIN = 2
learning_rate_MAX = 0.00009
learning_rate_MIN = 0.00001

vmax = []
vmin = []
vmax.append(n_MAX);vmax.append(alpha_MAX);vmax.append(learning_rate_MAX)
vmin.append(n_MIN);vmin.append(alpha_MIN);vmin.append(learning_rate_MIN)

# Loading Data
df = pd.read_csv("wifiData1.csv", header=0)

df = df[df['Local'] == 'IAD']

minMapeMPL1 = 100.0
maxR2 = 0

# Indexing the data
df['Date'] = pd.to_datetime(df['Date'])
#df.index = df['Date']
#del df['Date']
del df['Wlan']
del df['Local']
del df['Radio']

print(df)

t = 5
#df = df.resample('{}Min'.format(t))['Client'].count()
#df = df.rename(columns={'User' : 'Count'})
df = df.groupby([pd.Grouper(key='Date',freq='{}Min'.format(t))]).agg({'Client':'count'})
df = df.rename(columns={'Client' : 'count'})
print(df)


with open('Results/PSO_NN_PEMS.csv', 'w', 1) as nn_file:
	# Reading CSV
	nnwriter  = csv.writer(nn_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

	# Writing results headers
	nnwriter.writerow(['Particle','P', 'Q', 'N', 'Learning_Rate', 'Epoch' , 'Avg Mape', 'Min MAPE', 'Avg time'])
	
	df['count'] = df['count'] + 1
	
	# Using historic data (Q) from the same time and weekday
	for i in range (1, MAX_Q + 1):
		df['count-{}'.format(i)] = df['count'].shift(i * 60 * 24 // t)
	
	
	# Change n/a to 1
	df = df.fillna(1)

	# Normalizing the data
	df_max = max(df['count'])
	df_min = min(df['count'])
	df['count'] = df['count'] / (df_max - df_min)
	for i in range (1, MAX_Q + 1):
		df['count-{}'.format(i)] = df['count-{}'.format(i)] / (df_max - df_min)

	aux_df = df

	# Shifiting the data set by Q weeks
	df = df[q * 60 * 24 // t:]
	
	#df1 = df1[df1.index.weekday < 5]
	#df1 = df1.between_time('8:00','20:00')

	print('Running for params P = {}, Q = {}, N = {}'.format(p, q, n))
	print('Pre-processing...')

	# Initializing the data
	X1 = list()
	Y1 = list()

	# Mapping each set of variables (P and Q) to their correspondent value
	for i in range(len(df) - p - 1):
		X = list()
		for j in range (1, q + 1):
			X.append(df['count-{}'.format(j)][i + p + 1])

		X1.append(X + list(df['count'][i:(i + p)]))
		Y1.append(df['count'][i + p + 1])

	print('\nSplitting in train-test...')
		# Train/test/validation split
	rows1 = random.sample(range(len(X1)), int(len(X1)//3))

	X1_test = np.array( [X1[j] for j in rows1] )
	Y1_test = np.array( [Y1[j] for j in rows1] )
	X1_train = np.array( [X1[j] for j in list(set(range(len(X1))) - set(rows1))] )
	Y1_train = np.array([Y1[j] for j in list(set(range(len(Y1))) - set(rows1))] )

	avg_mlp_time1 = 0
	final_time = 0
	# Initializing the model
	X1_train = np.reshape(X1_train,(X1_train.shape[0], 1, X1_train.shape[1]))

	X1_test = np.reshape(X1_test,(X1_test.shape[0], 1, X1_test.shape[1]))
	shape = X1_train.shape[1:]
	print(shape)

	# Initializing the variables and the population
	pso = PSO(vmin, vmax)
	early_stopping_monitor = EarlyStopping(monitor='loss',patience=1)
	bestE = None
	bestCost = 0
	print('\nInitializing the population...')
	for i in range(population_size):
		pso.initializeParticle(i)
		param = pso.getPosition(i)
		MLP1 = ModelGRU(param[0], shape, epochsN, param[2], param[1])
		MLP1.fit(X1_train, Y1_train, epochs = epochsN,verbose=0, callbacks=[early_stopping_monitor])
		predicted1_nn = MLP1.predict(X1_test)
		print(Y1_test)
		print(predicted1_nn)
		try:
			cost = r2_score(Y1_test, predicted1_nn)
			pso.insertParticleCost(i,cost)
			if cost > bestCost:
				bestE = early_stopping_monitor.stopped_epoch
				bestCost = cost
		except:
			cost = 0
			pso.insertParticleCost(i,cost)
			if cost > bestCost:
				bestE = early_stopping_monitor.stopped_epoch
				bestCost = cost
		print(i)
		print("Epoch:" + str(early_stopping_monitor.stopped_epoch))
		MLP1.clearModel()

	pso.printGlobalBestParticle()
	nnwriter.writerow([pso.g_best_particle.id, p, q, pso.g_best_particle.position[0], pso.g_best_particle.position[2], bestE, pso.g_best_particle.p_best_cost, pso.g_best_particle.p_best_cost, avg_mlp_time1 / 30])

	iteration = 0
	print('\nRunning PSO Loop...')
	start_time = time.time()
	changesPSO = 0
	while(iteration < num_of_iterations):
		print('\nRunning... : {} of {}.'.format(iteration+1,num_of_iterations))
		for index in range(population_size):
			results_nn1 = list()
			print("Particle {}.".format(index))
			pso.population[index].calculateVelocity()
			pso.population[index].calculatePosition()

			n = int(pso.population[index].position[0])
			learning_rate = pso.population[index].position[2]
			alpha = pso.population[index].position[1]

			print('Running tests...')
			bestE = None
			bestCost = 0
			y = 1
			for test in range(testNumber):
				if((test + 1) % testNumber == (y)):
					if y + 1 == testNumber:
						y = 0
					else:
						y += 1
					print('T = {}%.'.format(((test + 1)/ testNumber) * 100))
				MLP1 = ModelGRU(n, shape, epochsN, learning_rate, alpha)
				MLP1.fit(X1_train, Y1_train, epochs = epochsN, verbose = 0, callbacks = [early_stopping_monitor])
				predicted1_nn = MLP1.predict(X1_test)
				try:
					currentCost = r2_score(Y1_test, predicted1_nn)

					results_nn1.append(currentCost)
					if(maxR2 < currentCost):
						trueValue = pd.DataFrame(Y1_test)
						bestMLP1value = pd.DataFrame(predicted1_nn)
						maxR2 = currentCost
						trueValue.to_csv("Results/TrueValue.csv")
						bestMLP1value.to_csv("Results/BestMLPvalue.csv")
				except:
					print('Error')

				MLP1.clearModel()

				if currentCost > bestCost:
					bestE = early_stopping_monitor.stopped_epoch
					bestCost = currentCost

			highestCurrentCost = max(results_nn1)
			if highestCurrentCost > pso.population[index].g_best_particle.p_best_cost:
				pso.insertParticleCost(i,highestCurrentCost)
				changesPSO += 1

			avg_pso_time1 = time.time() - start_time
			nnwriter.writerow([index, p, q, n, learning_rate, bestE, np.mean(results_nn1), max(results_nn1), avg_pso_time1])

		final_time = time.time() - start_time
		# print the best position, cost and particle of the population so far
		pso.printGlobalBestParticle()
		#print("Epoch = {}".format(pop[0].g_best_epoch))
		print("Time = {}".format(final_time))
		print("GBest_Change = {}".format(changesPSO))
		iteration += 1

	print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
df = aux_df
