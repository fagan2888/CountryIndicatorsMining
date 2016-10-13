import os
import re
import csv
import time
import json
import numpy as np
import scipy as sp
from decimal import Decimal
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

targetIDList = [2704,2707,2713,2716,2718,808,811,835]
# targetIDList = [2704]

def writeCSV(path,aList):
	"""
	This function dumps aList into a csv file as specified by path.

	Parameters
    ----------
    path : string
    	A filename or file path to save the csv file.

    aList : list
    	A list of data to write into the csv file.

    Return Value
    ----------
    None

	"""
	with open(path,'wb') as w:
		a = csv.writer(w, delimiter = ',')
		a.writerows(aList)
	w.close()

def getTargetIndexInData(targetName, dataName, targetIdx):
	"""
	This function finds the location of the focused target indicator in the list of predictor indicators. 

	Parameters
    ----------
    targetName : list
    	A list of target indicators name. 

    dataName : list
    	A list of predictor indicators name.

    targetIdx : int
    	An idex of focused target variable in the targetName list.

    Return Value
    ----------
    The index of the focused target indicator in the list of predictor indicators

	"""
	target = targetName[targetIdx]
	target = target[0:4]+target[5:]
	return dataName.index(target)

def getFeaturesIndex(predictorHeader,folderName,targetID,rankingMethod,numFeatures):
	"""
	This function finds all indices of predictors based on predictorHeader. 

	Parameters
    ----------
    predictorHeader : list
    	The list of header (from startPredictorIndex and so on).

    folderName : string
    	The name or path to the folder containing ranked predictor lists.

    targetID : int
    	The index (key) of the target variable.

    rankingMethod : int
    	The number specified how predictors are ranked. (See the file of ranked predictor lists)

    numFeatures : int
    	The number of features included in the model.

    Return Value
    ----------
    XIndex : all column indices of predictors included in the prediction model.

	"""

	f = open('./'+folderName+'/'+str(targetID)+'.csv','rb')
	reader = csv.reader(f)
	header = next(f)
	rows = list(reader)
	predictors = [row[rankingMethod] for row in rows]
	XIndex = []
	for idx in range(numFeatures):
		if predictors[idx] in predictorHeader:
			XIndex.append(predictorHeader.index(predictors[idx]))
	return XIndex

def getEstimator(algorithm, Yscaler, numFeatures):
	"""
	This function returns sklearn estimator as requested. 

	Parameters
    ----------
    algorithm : string (usually 3-character)
    	The code of prediction algorithm such as 'RID' for ridge regression.

    Yscaler : Scaler object of sklearn
    	Scaler object of sklearn fit to the training dataset

    numFeatures : int
    	The number of features included in the model.

    Return Value
    ----------
    sklearn estimator as requested

	"""

	halfFeatures = numFeatures/2+1
	estimator = None
	if algorithm == 'SVR':
		estimator = SVR(kernel='linear',epsilon=0.002*Yscaler.scale_)
	elif algorithm == 'SVL':
		estimator = LinearSVR(epsilon=0.002*Yscaler.scale_)
	elif algorithm == 'RBF':
		estimator = SVR(kernel='rbf',epsilon=0.002*Yscaler.scale_)
	elif algorithm == 'ELA':
		estimator = ElasticNet(alpha=0.4, l1_ratio=0.5)
	elif algorithm == 'LAS':
		estimator = Lasso(alpha = 0.2)
	elif algorithm == 'RID':
		estimator = Ridge(alpha = 0.2)
	elif algorithm == 'MLP':
		estimator = MLPRegressor()
	elif algorithm == 'ML1':
		estimator = MLPRegressor(hidden_layer_sizes = numFeatures)
	elif algorithm == 'ML2':
		estimator = MLPRegressor(hidden_layer_sizes = halfFeatures)
	elif algorithm == 'ML3':
		estimator = MLPRegressor(hidden_layer_sizes = numFeatures, alpha = 2.0/Yscaler.scale_)
	elif algorithm == 'ML4':
		estimator = MLPRegressor(hidden_layer_sizes = halfFeatures, alpha = 2.0/Yscaler.scale_)
	elif algorithm == 'ML5':
		estimator = MLPRegressor(hidden_layer_sizes = 1)
	elif algorithm == 'ML6':
		estimator = MLPRegressor(hidden_layer_sizes = 1, alpha = 2.0/Yscaler.scale_)
	elif algorithm == 'ML7':
		estimator = MLPRegressor(hidden_layer_sizes = [numFeatures]*2)
	elif algorithm == 'ML8':
		estimator = MLPRegressor(hidden_layer_sizes = [numFeatures]*3)
	elif algorithm == 'ML9':
		estimator = MLPRegressor(hidden_layer_sizes = [numFeatures]*10)
	return estimator

def getEstimatorGridSearch(algorithm, Yscaler, numFeatures):
	"""
	This function returns sklearn estimator with built-in grid search for parameter optimization as requested. 

	Parameters
    ----------
    algorithm : string (usually 3-character)
    	The code of prediction algorithm such as 'RID' for ridge regression.

    Yscaler : Scaler object of sklearn
    	Scaler object of sklearn fit to the training dataset

    numFeatures : int
    	The number of features included in the model.
    	
    Return Value
    ----------
    sklearn estimator with built-in grid search as requested

	"""

	halfFeatures = numFeatures/2+1
	estimator = None
	if algorithm == 'SVR':
		tuned_parameters = {'epsilon': [r*Yscaler.scale_ for r in [0.0005,0.001,0.005,0.01,0.05,0.1]], 'C':[0.01, 0.1, 1, 10, 100, 1000]}
		estimator = GridSearchCV(SVR(kernel='linear'), tuned_parameters, cv=3)
	elif algorithm == 'SVL':
		tuned_parameters = {'epsilon': [r*Yscaler.scale_ for r in [0.0005,0.001,0.005,0.01,0.05,0.1]], 'C':[0.01, 0.1, 1, 10, 100, 1000]}
		estimator = GridSearchCV(LinearSVR(), tuned_parameters, cv=3)
	elif algorithm == 'RBF':
		tuned_parameters = {'epsilon': [r*Yscaler.scale_ for r in [0.0005,0.001,0.005,0.01,0.05,0.1]], 'C':[0.01, 0.1, 1, 10, 100, 1000]}
		estimator = GridSearchCV(SVR(kernel='rbf'), tuned_parameters, cv=3)
	elif algorithm == 'ELA':
		estimator = ElasticNetCV(alphas=np.array([0.1, 0.5, 1.0, 5.0, 10.0]), l1_ratio=np.array([.1, .5, .7, .9, .95, .99, 1]), cv=3)
	elif algorithm == 'LAS':
		estimator = LassoCV(alphas=np.array([0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]), cv=3)
	elif algorithm == 'RID':
		estimator = RidgeCV(alphas=np.array([0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]), cv=3)
	else:
		tuned_parameters = [{'algorithm':['adam'], 'learning_rate':['constant'], 'max_iter':[200,500,1000], 'alpha':[1e-5,1e-4,1e-3]},
			{'algorithm':['sgd'], 'learning_rate':['adaptive'], 'max_iter':[200,500,1000], 'alpha':[1e-5,1e-4,1e-3]}]
		if algorithm == 'MLP':
			estimator = GridSearchCV(MLPRegressor(), tuned_parameters, cv=3)
		elif algorithm == 'ML1':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = numFeatures), tuned_parameters, cv=3)
		elif algorithm == 'ML2':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = halfFeatures), tuned_parameters, cv=3)
		elif algorithm == 'ML3':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = numFeatures, alpha = 2.0/Yscaler.scale_), tuned_parameters, cv=3)
		elif algorithm == 'ML4':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = halfFeatures, alpha = 2.0/Yscaler.scale_), tuned_parameters, cv=3)
		elif algorithm == 'ML5':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = 1), tuned_parameters, cv=3)
		elif algorithm == 'ML6':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = 1, alpha = 2.0/Yscaler.scale_), tuned_parameters, cv=3)
		elif algorithm == 'ML7':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = [numFeatures]*2), tuned_parameters, cv=3)
		elif algorithm == 'ML8':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = [numFeatures]*3), tuned_parameters, cv=3)
		elif algorithm == 'ML9':
			estimator = GridSearchCV(MLPRegressor(hidden_layer_sizes = [numFeatures]*10), tuned_parameters, cv=3)
	return estimator

def crossValidation(filename):
	"""
	This function evaluates the performance of all possible algorithms to predict target variable in the next year
	using leave-one-out cross validation and mean absolute value.

	This function is used in iteration 3.

	** This function is hard coded. Please be careful while editing. **

	Parameters
    ----------
    filename : string
    	The name or path of the json file used in iteration 3.

    Return Value
    ----------
    None, but the results are saved as files instead.

	"""

	f = open(filename, 'rb')
	allData = json.load(f)

	collectBest = [['targetIdx', 'rankingMethod', 'algorithm', 'numFeatures', 'score', 'sd', 'timeProcessed']]

	for targetIdx in range(len(targetIDList)):
		rows = []
		for rankingMethod in [0,1]:		

			numFeaturesTest = [1,2,3,4,5,10,15,20,25,30,35,40,45,50]
			# numFeaturesTest = range(1,51)
			# algos = ['SVL','RBF', 'LAS', 'RID', 'ELA', 'MLP','ML3','ML4','ML7','ML8','ML9']	
			algos = ['SVL','RBF', 'RID', 'ELA', 'MLP','ML3','ML4','ML8','ML9']	
			# numFeaturesTest = range(1,3)
			# algos = ['SVL']	

			best = [targetIdx, rankingMethod, None,None,None,None,None]
			
			scoreTable = [numFeaturesTest]
			timeTable = [numFeaturesTest]
			sdTable = [numFeaturesTest]

			for algorithm in algos:

				scoreList = []
				timeList = []
				sdList = []
				
				for numFeatures in numFeaturesTest:
					error = []
					startTime = time.time()
					for key, countryObject in allData.iteritems():
						target = np.array(countryObject['target'])
						data = np.array(countryObject['data'])

						X = data[:-1,:]
						y = target[:-1,targetIdx]

						keepRows = np.array([True if x != '' else False for x in y])
						if keepRows.sum() < 0.5*keepRows.shape[0]:
							continue
						
						XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod,numFeatures)
						Xready = X[keepRows,:]
						Xready = Xready[:,tuple(XIndex)]
						y = y[keepRows].astype(np.float)						
						y = np.ravel(y)
						y = y.reshape(-1,1)
						if len(XIndex) == 1:
							Xready = Xready.reshape(-1,1)
						
						Xscaler = preprocessing.StandardScaler().fit(Xready)
						Xscaler.transform(Xready)
						Yscaler = preprocessing.StandardScaler().fit(y)
						
						
						estimator = getEstimator(algorithm,Yscaler,numFeatures)
						if estimator is None:
							return algorithm + ': Wrong Algorithm'

						
						crossValScoreList = (-1) * cross_val_score(estimator, Xready, np.ravel(y), cv=Xready.shape[0], scoring='mean_absolute_error')
						error.extend(crossValScoreList.tolist())

					score = np.array(error).mean()
					sd = np.array(error).std()
					timeProcessed = time.time() - startTime

					scoreList.append(score)
					sdList.append(sd)
					timeList.append(timeProcessed)

					if best[2] is None:
						best = [targetIdx, rankingMethod, algorithm, numFeatures, score, sd, timeProcessed]
					else:
						if score < best[4]:
							best = [targetIdx, rankingMethod, algorithm, numFeatures, score, sd, timeProcessed]
					# break
				scoreTable.append(scoreList)
				timeTable.append(timeList)
				sdTable.append(sdList)
			
			print 'target %d, rankingMethod %d' % (targetIdx, rankingMethod)
			print best
			collectBest.append(best)

			scoreTable = np.transpose(np.array(scoreTable)).tolist()
			timeTable = np.transpose(np.array(timeTable)).tolist()
			sdTable = np.transpose(np.array(sdTable)).tolist()

			rows.append(['score of ranking method = '+str(rankingMethod)]+algos) 
			rows.extend(scoreTable)
			rows.append(['sd of ranking method = '+str(rankingMethod)]+algos) 
			rows.extend(sdTable)
			rows.append(['time of ranking method = '+str(rankingMethod)]+algos) 
			rows.extend(timeTable)
			rows.append((1+len(algos))*[''])

		writeCSV('./Country Validation 2006-2012/Indicator'+str(targetIDList[targetIdx])+'-'+time.strftime("%Y-%m-%d-%H-%M-%S")+'-'.join(algos)+'.csv',rows)
		writeCSV('./Country Validation 2006-2012/CollectBest'+'-'+time.strftime("%Y-%m-%d-%H-%M-%S")+'.csv',collectBest)

def testModel(filename, targetIdx, rankingMethod, algorithm, numFeatures, bootstrapping = False, gridSearch = False, estimator = None):
	"""
	This function trains and tests the input model configurations for the specified target variable.

	This function is used in iteration 3 and the resuls are averaged from all country objects.

	** This function is hard coded. Please be careful while editing. **

	Parameters
    ----------
    filename : string
    	The name of json file in iteration 3.

    targetIdx : int
    	The index of target variable in the target list.

    rankingMethod : int
    	The number specified how predictors are ranked. (See the file of ranked predictor lists)

    algorithm : string (usually 3-character)
    	The code of prediction algorithm such as 'RID' for ridge regression.

    numFeatures : int
    	The number of features included in the model.

    bootstrapping : boolean
   		Perform bootstrapping prediction or not.

    gridSearch : boolean
    	Perform parameter optimization or not.

    estimator : sklearn estimator
    	An estimator to test (optional since it can be indicated by other parameters).

    Return Value
    ----------
    None, but the results (error, sd) are printed.

	"""
	f = open(filename, 'rb')
	allData = json.load(f)
	error = []

	for key, countryObject in allData.iteritems():
		target = np.array(countryObject['target'])
		data = np.array(countryObject['data'])

		historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
		targetY = target[:,targetIdx]
		answer = targetY[-1]

		if answer == '':
			continue
		else:
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = rankingMethod,numFeatures = numFeatures)
			keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
			if keepRows.sum() <= 2:
				continue
			X = data[keepRows,:]
			X = X[:, tuple(XIndex)]
			y = targetY[keepRows].astype(np.float)
			y = y.reshape(-1,1)
			
			if len(XIndex) == 1:
				X = X.reshape(-1,1)
			Xscaler = preprocessing.StandardScaler().fit(X)
			Xscaler.transform(X)
			Yscaler = preprocessing.StandardScaler().fit(y)
			
			if estimator is None:
				if not gridSearch:
					estimator = getEstimator(algorithm,Yscaler,numFeatures)
				else:
					estimator = getEstimatorGridSearch(algorithm,Yscaler,numFeatures)

			if estimator is None:
				return algorithm + ': Wrong Algorithm'

			XTest = data[-1, tuple(XIndex)]
			XTest = XTest.reshape(1,-1)
			Xscaler.transform(XTest)

			if not bootstrapping:
				estimator.fit(X,y)
				predicted = estimator.predict(XTest)
			else:
				predicted = bootstrappingPrediction(estimator, X, y, XTest)
			
			error.append(np.absolute(float(answer) - predicted))

	score_mean = np.array(error).mean()
	score_sd = np.array(error).std()
	
	print '%s = rankingMethod %d, algo %s, numFeatures %d, score(mean, sd) = (%f,%f)' % (targetIDList[targetIdx], rankingMethod, algorithm, numFeatures, score_mean, score_sd)

def bootstrappingPrediction(estimator, Xready, y, XTestReady, B = 200, parametric = True):
	"""
	This function performs bootstrapping prediction instead of normal prediction.

	Parameters
    ----------
    estimator : sklearn estimator
    	An estimator object used to predict the target value.

    Xready : 2D numpy array
    	An array of training instances.

    y : 1D numpy array
    	An array of target values for training instances.

    XTestReady : 2D numpy array
    	An array of testing instances.

    B : int
    	The number of bootstrapping models

    parametric : boolean
    	(1 = Parametric bootstrapping - normal distribution) VS (0 = Nonparametric bootstrapping)

    Return Value
    ----------
    1D numpy array of prediction results corresponding to test instances.

	"""
	estimator.fit(Xready, y)
	predicted = estimator.predict(Xready)
	error = y - predicted
	numExamples = len(predicted)

	testPredicted = np.array([0]*XTestReady.shape[0])
	
	for i in range(B):
		if parametric:
			if error.std() == 0:
				noise = np.random.normal(error.mean(),0.0001,numExamples)
			else:
				noise = np.random.normal(error.mean(),error.std(),numExamples)
		else:
			noise = np.array([error[np.random.randint(0, numExamples)] for i in range(numExamples)])
		
		newY = predicted + noise
		estimator.fit(Xready, newY)
		testPredicted = testPredicted + estimator.predict(XTestReady)

	return testPredicted / np.array([B*1.0])

# crossValidation('New_2006-2013_data.json')
# options = [
# (0,	0,	'ELA',	1),(0,	1,	'ELA',	1),
# (1,	0,	'ELA',	1),(1,	1,	'ELA',	1),
# (2,	0,	'RID',	2),(2,	1,	'RID',	1),
# (3,	0,	'RBF',	1),(3,	1,	'ELA',	15),
# (4,	0,	'RBF',	1),(4,	1,	'RBF',	1),
# (5,	0,	'ELA',	1),(5,	1,	'ELA',	1),
# (6,	0,	'RBF',	2),(6,	1,	'RBF',	3),
# (7,	0,	'SVL',	40),(7,	1,	'ELA',	1)]

# print 'No bootstrapping + GridSearch'
# for op in options:
# 	testModel('New_2006-2013_data.json',op[0],op[1],op[2],op[3], bootstrapping = False, gridSearch = True)

# print 'Bootstrapping + GridSearch'
# for op in options:
# 	testModel('New_2006-2013_data.json',op[0],op[1],op[2],op[3], bootstrapping = True, gridSearch = True)

options = [
(0,	0,	'ELA',	1, ElasticNet(alpha=0.1, l1_ratio=0.1)),
(0,	1,	'ELA',	1, ElasticNet(alpha=0.1, l1_ratio=0.1)),
(1,	0,	'ELA',	1, ElasticNet(alpha=0.1, l1_ratio=0.1)),
(1,	1,	'ELA',	1, ElasticNet(alpha=0.1, l1_ratio=0.1)),
(2,	0,	'RID',	2, Ridge(alpha = 0.5)),
(2,	1,	'RID',	1, Ridge(alpha = 0.5)),
(3,	0,	'RBF',	1, SVR(kernel='rbf',epsilon=0.0119298, C=1000)),
(3,	1,	'ELA',	15, ElasticNet(alpha=0.1, l1_ratio=0.5)),
(4,	0,	'RBF',	1, SVR(kernel='rbf',epsilon=0.21971793, C=100)),
(4,	1,	'RBF',	1, SVR(kernel='rbf',epsilon=0.21971793, C=100)),
(5,	0,	'ELA',	1, ElasticNet(alpha=10, l1_ratio=0.1)),
(5,	1,	'ELA',	1, ElasticNet(alpha=10, l1_ratio=0.1)),
(6,	0,	'RBF',	2, SVR(kernel='rbf',epsilon=0.54155267, C=1)),
(6,	1,	'RBF',	3, SVR(kernel='rbf',epsilon=0.05415527, C=1)),
(7,	0,	'SVL',	40, LinearSVR(epsilon=900.56680603, C=1)),
(7,	1,	'ELA',	1, ElasticNet(alpha=0.1, l1_ratio=0.1))]

print 'No bootstrapping + GridAlreadySearch General Param Op'
for op in options:
	testModel('New_2006-2013_data.json',op[0],op[1],op[2],op[3], bootstrapping = False, gridSearch = False, estimator = op[4])

print 'Bootstrapping + GridAlreadySearch General Param Op'
for op in options:
	testModel('New_2006-2013_data.json',op[0],op[1],op[2],op[3], bootstrapping = True, gridSearch = False, estimator = op[4])