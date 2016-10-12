import os
import re
import csv
import time
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.neural_network import MLPRegressor
# from sknn.mlp import Regressor, Layer
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import *
from pprint import pprint
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


# targetList = [2704,2707,2713,2716,2718,808,811,1954]
targetList = [2704,2707,2713,2716,2718,808,811,835]
# targetList = [1994,1997,2003,2006,2008,807,810,1953]
yearList = [(1998,2015),(2005,2015),(2005,2015),(2005,2015),(2005,2015),(1960,2014),(1961,2014),(2002,2012)]

def writeCSV(path,aList):
	with open(path,'wb') as w:
		a = csv.writer(w, delimiter = ',')
		a.writerows(aList)
	w.close()

def getHeader(filepath):
	with open(filepath, 'rb') as f:
		reader = csv.reader(f)
		header = next(reader)
		f.close()
	return header

def getIndicatorRank(targetIndex, allFeatures = 0, toNextYear = 0, rankMethod = 0): 
	# allFeatures = 0 # All VS Ext
	# toNextYear = 0 # N VS P
	# rankMethod = 0 # MIC VS FST
	col = allFeatures * 4 + toNextYear * 2 + rankMethod * 1

	indicatorRank = []

	with open('./Feature Ranking/'+str(targetIndex)+'.csv','rb') as f:
		reader = csv.reader(f)
		countRow = 0
		for row in reader:
			if countRow == 0:
				countRow += 1
			else:
				if row[col] != '':
					indicatorRank.append(row[col])
				countRow += 1
		f.close()

	return indicatorRank

def findFileName(folderName, year):
	for root, dirs, files in os.walk('./'+folderName):    
	    for afile in files:
	    	if afile.startswith(str(year)) and len(afile) == 28:
	    		return afile
	return 'Not found'

def getColumnIndexList(folderName, filename, targetIndex, indicatorRank):

	columnIndexList = []
	with open('./'+folderName+'/'+filename,'rb') as f:
		reader = csv.reader(f)
		header = next(reader)

		yIndex = [idx for idx, item in enumerate(header) if item.startswith(str(targetIndex).zfill(4)+'N')][0]

		for desRank in indicatorRank:
			for idx, item in enumerate(header):
				if item.startswith(desRank):
					columnIndexList.append(idx)
					break

		f.close()

	return columnIndexList, yIndex

def shapingDataset(folderName, year, targetIndex, indicatorRank):
	filename = findFileName(folderName, year)
	XIndex, yIndex = getColumnIndexList(folderName, filename, targetIndex, indicatorRank)
	dataset = np.genfromtxt('./'+folderName+'/'+filename, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan)
	X = dataset[:,tuple(XIndex)]
	y = dataset[:,yIndex]
	# print X.shape
	# print y.shape
	# kill missing target rows
	keepRows = np.invert(np.isnan(y))
	# print keepRows.shape
	X = X[keepRows,:]
	y = y[keepRows]
	return X, y

def crossValidation(targetIndexInList, allFeatures = 0, toNextYear = 0, rankMethod = 0, numFeatures = 50, algorithm = 'SVR', cv=5):
	# allFeatures = 0 # All VS Ext
	# toNextYear = 0 # N VS P
	# rankMethod = 0 # MIC VS FST

	targetIndex = targetList[targetIndexInList]
	indicatorRank = getIndicatorRank(targetIndex, allFeatures, toNextYear, rankMethod)
	if numFeatures <= len(indicatorRank):
		indicatorRank = indicatorRank[:numFeatures]
	
	if toNextYear == 0:
		folderName = 'FormattedFilesWithoutMissingToNextYear'
	else:
		folderName = 'Formatted Files Without Missing'

	rows = [['Year','Mean Absolute Error','Time Spent (s)','Error for each fold']]
	scoreEachFold = []
	timeEachFold = []
	for year in range(yearList[targetIndexInList][0]-1,yearList[targetIndexInList][1]):
		startTime = time.time()
		X, y = shapingDataset(folderName, year, targetIndex, indicatorRank)
		if X.shape[1] == 0:
			continue
			# print 'Hey'
		estimator = None
		if algorithm == 'SVR':
			estimator = Pipeline([("imputer", Imputer(missing_values='NaN', strategy='median', axis=0)), ("svr", SVR(kernel='linear'))])
		elif algorithm == 'RBF':
			estimator = Pipeline([("imputer", Imputer(missing_values='NaN', strategy='median', axis=0)), ("rbf", SVR(kernel='rbf'))])
		elif algorithm == 'ELA':
			estimator = Pipeline([("imputer", Imputer(missing_values='NaN', strategy='median', axis=0)), ("ela", ElasticNet(alpha=0.4, l1_ratio=0.5, normalize=False))])
		elif algorithm == 'LAS':
			estimator = Pipeline([("imputer", Imputer(missing_values='NaN', strategy='median', axis=0)), ("las", Lasso(alpha = 0.2, normalize = True))])
		elif algorithm == 'RID':
			estimator = Pipeline([("imputer", Imputer(missing_values='NaN', strategy='median', axis=0)), ("rid", Ridge(alpha = 0.2, normalize = True))])
		else:
			return 'Wrong Algorithm' , 'Wrong Algorithm'
		scoreList = cross_val_score(estimator, X, y, cv=cv, scoring='mean_absolute_error')
		score = scoreList.mean()
		# print("Score after imputation of the missing values = %.2f" % score)
		scoreEachFold.append(score)
		timecount = time.time() - startTime
		timeEachFold.append(timecount)
		rows.append([year,score,timecount,scoreList])
		# print year

	averageScore = np.array(scoreEachFold).mean()
	sumTime = np.array(timeEachFold).sum()

	filename = './CrossValidation/'+('Indicator%d,allFeatures=%d,toNextYear=%d,rankMethod=%d,numFeatures=%d,algorithm=%s,cv=%d.csv' % (targetIndex,allFeatures,toNextYear,rankMethod,numFeatures,algorithm,cv))
	with open(filename,'wb') as w:
		a = csv.writer(w, delimiter = ',')
		a.writerows(rows)
	w.close()
	print ('algo=%s,nf=%d,allFeatures=%d,rankMethod=%d,averageScore=%f,sumTime=%f' % (algorithm,numFeatures,allFeatures,rankMethod,averageScore,sumTime))
	return averageScore, sumTime

def getTargetAndPredictorIndex(header):
	startTargetIndex = None
	startPredictorIndex = None
	numCols = len(header)
	regex = re.compile("....N:.*")
	idx = 0
	for col in header:
		if col.startswith('Year'):
			pass
		elif regex.search(col) and startTargetIndex is None:
			startTargetIndex = idx
		elif not regex.search(col) and not startTargetIndex is None and startPredictorIndex is None:
			startPredictorIndex = idx
			break
		idx += 1
	return startTargetIndex, startPredictorIndex, numCols

def getFeaturesIndex(predictorHeader,folderName,targetID,rankingMethod,numFeatures):
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

def crossValidationLargeFile(filename, cv=5):
	# filename = '2006-2013_FilteredColsTargetMissingBlank.csv'
	header = getHeader(filename)
	startTargetIndex, startPredictorIndex, numCols = getTargetAndPredictorIndex(header)
	numTargets = startPredictorIndex - startTargetIndex
	numPredictors = numCols - startPredictorIndex
	predictorHeader = header[startPredictorIndex:]
	targetHeader = header[startTargetIndex:startPredictorIndex]
	targetIDList = [int(head[0:4]) for head in targetHeader]
	
	collectBest = [['targetIdx', 'rankingMethod', 'algorithm', 'numFeatures', 'score', 'sd', 'timeProcessed']]

	dataset = np.genfromtxt(filename, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(range(startTargetIndex,numCols)))
	for targetIdx in range(0,numTargets):
		X = dataset[:,tuple(range(numTargets,dataset.shape[1]))]
		y = dataset[:,targetIdx]

		keepRows = np.invert(np.isnan(y))
		X = X[keepRows,:]
		y = y[keepRows]
		y = y.reshape(-1,1)

		Xscaler = preprocessing.StandardScaler().fit(X)
		Xscaler.transform(X)

		Yscaler = preprocessing.StandardScaler().fit(y)
		# Yscaler.transform(y)
		algos = ['SVL','RBF', 'LAS', 'RID', 'ELA', 'MLP','ML1','ML2','ML3','ML4','ML5','ML6','ML7','ML8','ML9']	
		print 'Target %d => Mean %.5f , STD %.5f, Min %.5f, Max %.5f' % (targetIdx, Yscaler.mean_, Yscaler.scale_, y.min(), y.max())
		rows = []
		for rankingMethod in [0,1,2,3]:		
			# numFeaturesTest = [5]
			numFeaturesTest = range(1,51)

			best = [targetIdx, rankingMethod, None,None,None,None,None]
			scoreTable = [numFeaturesTest]
			timeTable = [numFeaturesTest]
			sdTable = [numFeaturesTest]

			for algorithm in algos:

				scoreList = []
				timeList = []
				sdList = []
				
				for numFeatures in numFeaturesTest:
					estimator = getEstimator(algorithm,Yscaler,numFeatures)
					if estimator is None:
						return algorithm + ': Wrong Algorithm'
					XIndex = getFeaturesIndex(predictorHeader,'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod,numFeatures)
					Xready = X[:,tuple(XIndex)]
					y = np.ravel(y)

					startTime = time.time()
					# 1)
					# crossValScoreList = cross_val_score(estimator, Xready, y, cv=cv, scoring='mean_absolute_error')
					# score = crossValScoreList.mean()

					# 2)
					# predicted = cross_val_predict(estimator, Xready, y, cv=cv)
					# absolute_error = np.absolute(y - predicted)
					# score = -absolute_error.mean()
					# sd = absolute_error.std()

					# 3)
					allCases = Xready.shape[0]
					firstTestIndex = int(0.75*Xready.shape[0])
					estimator.fit(Xready[0:firstTestIndex,:],y[0:firstTestIndex])
					predicted = estimator.predict(Xready[firstTestIndex:,:])
					absolute_error = np.absolute(y[firstTestIndex:] - predicted)
					score = -absolute_error.mean()
					sd = absolute_error.std()

					timeProcessed = time.time()-startTime
					scoreList.append(-score)
					sdList.append(sd)
					timeList.append(timeProcessed)
					# print targetIdx, rankingMethod, algorithm, numFeatures, score, sd, timeProcessed
					if best[2] is None:
						best = [targetIdx, rankingMethod, algorithm, numFeatures, -score, sd, timeProcessed]
					else:
						if -score < best[4]:
							best = [targetIdx, rankingMethod, algorithm, numFeatures, -score, sd, timeProcessed]
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

		writeCSV('./Validation 2006-2014/Indicator'+str(targetIDList[targetIdx])+'-'+time.strftime("%Y-%m-%d-%H-%M-%S")+'-'.join(algos)+'.csv',rows)
	writeCSV('./Validation 2006-2014/CollectBest'+'-'+time.strftime("%Y-%m-%d-%H-%M-%S")+'.csv',collectBest)
		# break

def validationLargeFile(filename, testFilename, cv=5):
	# filename = '2006-2013_FilteredColsTargetMissingBlank.csv'
	header = getHeader(filename)
	startTargetIndex, startPredictorIndex, numCols = getTargetAndPredictorIndex(header)
	numTargets = startPredictorIndex - startTargetIndex
	numPredictors = numCols - startPredictorIndex
	predictorHeader = header[startPredictorIndex:]
	targetHeader = header[startTargetIndex:startPredictorIndex]
	targetIDList = [int(head[0:4]) for head in targetHeader]
	
	collectBest = [['targetIdx', 'rankingMethod', 'algorithm', 'numFeatures', 'score', 'sd', 'timeProcessed']]

	dataset = np.genfromtxt(filename, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(range(startTargetIndex,numCols)))
	testset = np.genfromtxt(testFilename, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(range(startTargetIndex,numCols)))

	for targetIdx in range(0,numTargets):
		# Training Data ---------------------------------------------------
		X = dataset[:,tuple(range(numTargets,dataset.shape[1]))]
		y = dataset[:,targetIdx]

		keepRows = np.invert(np.isnan(y))
		X = X[keepRows,:]
		y = y[keepRows]
		y = y.reshape(-1,1)

		Xscaler = preprocessing.StandardScaler().fit(X)
		Xscaler.transform(X)

		# Test Data ---------------------------------------------------
		XTest = testset[:,tuple(range(numTargets,testset.shape[1]))]
		yTest = testset[:,targetIdx]

		keepRowsTest = np.invert(np.isnan(yTest))
		XTest = XTest[keepRowsTest,:]
		yTest = yTest[keepRowsTest]
		yTest = yTest.reshape(-1,1)
		Xscaler.transform(XTest)


		Yscaler = preprocessing.StandardScaler().fit(np.concatenate((y,yTest)))
		# Yscaler.transform(y)

		algos = ['SVL','RBF', 'LAS', 'RID', 'ELA', 'MLP','ML1','ML2','ML3','ML4','ML5','ML6','ML7','ML8','ML9']	
		print 'Target %d => Mean %.5f , STD %.5f, Min %.5f, Max %.5f' % (targetIdx, Yscaler.mean_, Yscaler.scale_, np.concatenate((y,yTest)).min(), np.concatenate((y,yTest)).max())
		Yscaler = preprocessing.StandardScaler().fit(y)
		
		# continue
		rows = []
		for rankingMethod in [0,1,2,3]:		
			# numFeaturesTest = [5]
			numFeaturesTest = range(1,51)

			best = [targetIdx, rankingMethod, None,None,None,None,None]
			scoreTable = [numFeaturesTest]
			timeTable = [numFeaturesTest]
			sdTable = [numFeaturesTest]

			for algorithm in algos:

				scoreList = []
				timeList = []
				sdList = []
				
				for numFeatures in numFeaturesTest:
					estimator = getEstimator(algorithm,Yscaler,numFeatures)
					if estimator is None:
						return algorithm + ': Wrong Algorithm'
					XIndex = getFeaturesIndex(predictorHeader,'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod,numFeatures)
					Xready = X[:,tuple(XIndex)]
					y = np.ravel(y)

					startTime = time.time()

					# estimator.fit(Xready,y)
					# XTestReady = XTest[:,tuple(XIndex)]
					# predicted = estimator.predict(XTestReady)
					# absolute_error = np.absolute(yTest - predicted)


					firstTestIndex = int(0.75*Xready.shape[0])
					estimator.fit(Xready[0:firstTestIndex,:],y[0:firstTestIndex])
					predicted = estimator.predict(Xready[firstTestIndex:,:])
					absolute_error = np.absolute(y[firstTestIndex:] - predicted)
					
				
					score = absolute_error.mean()
					sd = absolute_error.std()

					timeProcessed = time.time()-startTime
					scoreList.append(score)
					sdList.append(sd)
					timeList.append(timeProcessed)
					# print targetIdx, rankingMethod, algorithm, numFeatures, score, sd, timeProcessed
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

		writeCSV('./Validation 2006-2012 Three Set/Indicator'+str(targetIDList[targetIdx])+'-'+time.strftime("%Y-%m-%d-%H-%M-%S")+'-'.join(algos)+'.csv',rows)
		writeCSV('./Validation 2006-2012 Three Set/CollectBest'+'-'+time.strftime("%Y-%m-%d-%H-%M-%S")+'.csv',collectBest)

def crossValidationOneOption(filename, targetIdx, rankingMethod, algorithm, numFeatures, cv=5):
		# filename = '2006-2013_FilteredColsTargetMissingBlank.csv'
	header = getHeader(filename)
	startTargetIndex, startPredictorIndex, numCols = getTargetAndPredictorIndex(header)
	numTargets = startPredictorIndex - startTargetIndex
	numPredictors = numCols - startPredictorIndex
	predictorHeader = header[startPredictorIndex:]
	targetHeader = header[startTargetIndex:startPredictorIndex]
	targetIDList = [int(head[0:4]) for head in targetHeader]

	dataset = np.genfromtxt(filename, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(range(startTargetIndex,numCols)))
	# for targetIdx in range(0,numTargets):
	X = dataset[:,tuple(range(numTargets,dataset.shape[1]))]
	y = dataset[:,targetIdx]

	keepRows = np.invert(np.isnan(y))
	X = X[keepRows,:]
	y = y[keepRows]
	y = y.reshape(-1,1)

	Xscaler = preprocessing.StandardScaler().fit(X)
	Xscaler.transform(X)

	Yscaler = preprocessing.StandardScaler().fit(y)
	# Yscaler.transform(y)

	# print 'Target %d => Mean %.5f , STD %.5f' % (targetIdx, Yscaler.mean_, Yscaler.scale_)
	estimator = getEstimator(algorithm,Yscaler,numFeatures)
	if estimator is None:
		return algorithm + ': Wrong Algorithm'	
	
	XIndex = getFeaturesIndex(predictorHeader,'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod,numFeatures)
	Xready = X[:,tuple(XIndex)]
	y = np.ravel(y)

	startTime = time.time()
	# crossValScoreList = cross_val_score(estimator, Xready, y, cv=cv, scoring='mean_absolute_error')
	predicted = cross_val_predict(estimator, Xready, y, cv=cv)
	absolute_error = np.absolute(y - predicted)
	score_mean = absolute_error.mean()
	score_sd = absolute_error.std()
	timeProcessed = time.time()-startTime
	
	print '%s = rankingMethod %d, algo %s, numFeatures %d, score(mean, sd) = (%f,%f), time = %f' % (targetHeader[targetIdx], rankingMethod, algorithm, numFeatures, score_mean, score_sd, timeProcessed)

def getSimilarColIndex(header, testHeader):
	indexList = []
	for i in header:
		if i in testHeader[1:]:
			indexList.append(testHeader.index(i, 1))
	return indexList

def testModel(testFilename, filename, targetIdx, rankingMethod, algorithm, numFeatures, gridSearch = False,cv=5):
		# filename = '2006-2013_FilteredColsTargetMissingBlank.csv'
	header = getHeader(filename)
	startTargetIndex, startPredictorIndex, numCols = getTargetAndPredictorIndex(header)
	numTargets = startPredictorIndex - startTargetIndex
	numPredictors = numCols - startPredictorIndex
	predictorHeader = header[startPredictorIndex:]
	targetHeader = header[startTargetIndex:startPredictorIndex]
	targetIDList = [int(head[0:4]) for head in targetHeader]

	dataset = np.genfromtxt(filename, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(range(startTargetIndex,numCols)))
	# for targetIdx in range(0,numTargets):
	X = dataset[:,tuple(range(numTargets,dataset.shape[1]))]
	y = dataset[:,targetIdx]

	keepRows = np.invert(np.isnan(y))
	X = X[keepRows,:]
	y = y[keepRows]
	y = y.reshape(-1,1)

	print X.shape
	Xscaler = preprocessing.StandardScaler().fit(X)
	Xscaler.transform(X)

	Yscaler = preprocessing.StandardScaler().fit(y)
	# Yscaler.transform(y)

	# print 'Target %d => Mean %.5f , STD %.5f' % (targetIdx, Yscaler.mean_, Yscaler.scale_)
	if not gridSearch:
		estimator = getEstimator(algorithm,Yscaler,numFeatures)
	else:
		estimator = getEstimatorGridSearch(algorithm,Yscaler,numFeatures)
	if estimator is None:
		return algorithm + ': Wrong Algorithm'	
	
	XIndex = getFeaturesIndex(predictorHeader,'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod,numFeatures)
	Xready = X[:,tuple(XIndex)]
	y = np.ravel(y)

	estimator.fit(Xready,y)
	# estimator.fit(Xready[0:int(0.75*Xready.shape[0]),:],y[0:int(0.75*Xready.shape[0])])
	print 'Best params for %s = rankingMethod %d, algo %s, numFeatures %d' % (targetHeader[targetIdx][0:4], rankingMethod, algorithm, numFeatures)
	if hasattr(estimator, 'best_params_'):
		print estimator.best_params_
	else:
		pprint(vars(estimator)) 

	#---------------------------------------Test-------------------------------------------------------------------------------------------
	testHeader = getHeader(testFilename)
	filteredCols = getSimilarColIndex(header, testHeader)
	# print filteredCols
	if numCols != len(filteredCols) + 1:
		print 'Column Error'
		return 'Column Error'
	
	testset = np.genfromtxt(testFilename, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(filteredCols))
	XTest = testset[:,tuple(range(numTargets,testset.shape[1]))]
	yTest = testset[:,targetIdx]
	
	haveNotNull = np.sum(np.invert(np.isnan(XTest)),axis=0)
	missing = []
	lowDense = []
	for col in range(len(haveNotNull)):
		if haveNotNull[col] == 0:
			XTest[:,col] = np.array([Xscaler.mean_[col]]*XTest.shape[0])
			missing.append(col)
		if haveNotNull[col] < 0.4*XTest.shape[0]:
			lowDense.append(col)
	print 'Low dense = %d' % (len(lowDense))

	imp = Imputer(missing_values='NaN', strategy='median', axis=0)
	XTest = imp.fit_transform(XTest)

	keepRows = np.invert(np.isnan(yTest))
	XTest = XTest[keepRows,:]
	yTest = yTest[keepRows]
	yTest = yTest.reshape(-1,1)

	Xscaler.transform(XTest)
	XTestReady = XTest[:,tuple(XIndex)]

	# predicted = estimator.predict(XTestReady)
	predicted = bootstrappingPrediction(estimator, Xready, y, XTestReady, B = 200, parametric = False)
	absolute_error = np.absolute(yTest - predicted)
	# predicted = estimator.predict(Xready[int(0.75*Xready.shape[0]):,:])
	# absolute_error = np.absolute(y[int(0.75*Xready.shape[0]):] - predicted)

	score_mean = absolute_error.mean()
	score_sd = absolute_error.std()
	
	missingCount = len(list(set(missing) & set(XIndex)))
	lowDenseCount = len(list(set(lowDense) & set(XIndex)))
	# print score_mean, score_sd
	print '%s = rankingMethod %d, algo %s, numFeatures %d, score(mean, sd) = (%f,%f), missing = %d, lowDense = %d' % (targetHeader[targetIdx][0:4], rankingMethod, algorithm, numFeatures, score_mean, score_sd, missingCount, lowDenseCount)

def bootstrappingPrediction(estimator, Xready, y, XTestReady, B = 200, parametric = True):
	estimator.fit(Xready, y)
	predicted = estimator.predict(Xready)
	error = y - predicted
	numExamples = len(predicted)

	testPredicted = np.array([0]*XTestReady.shape[0])
	
	for i in range(B):
		if parametric:
			noise = np.random.normal(error.mean(),error.std(),numExamples)
		else:
			noise = np.array([error[np.random.randint(0, numExamples)] for i in range(numExamples)])
		
		newY = predicted + noise
		estimator.fit(Xready, newY)
		testPredicted = testPredicted + estimator.predict(XTestReady)

	return testPredicted / np.array([B*1.0])

# -------------------------------------------------------------------------------------------------------
# options = [(0,0,'RID',1),(0,1,'RID',1),(0,3,'RID',50),
# (1,1,'RID',5),(1,2,'LAS',47),
# (2,1,'SVL',1),(2,2,'RID',46),
# (3,0,'SVL',1),(3,3,'RID',46),
# (4,0,'RBF',1),(4,1,'RBF',1),(4,2,'RID',50),
# (5,0,'SVL',2),(5,3,'SVL',36),
# (6,0,'RBF',1),(6,3,'RID',21),
# (7,0,'RID',2),(7,3,'SVL',26)
# ]

# for op in options:
# 	crossValidationOneOption('2006-2013_FilteredColsTargetMissingBlank.csv', op[0], op[1], op[2], op[3])
# -------------------------------------------------------------------------------------------------------

# crossValidationLargeFile('New_2006-2013_FilteredColsTargetMissingBlank.csv')
# crossValidationLargeFile('Train_2006-2014.csv')

# validationLargeFile('1-Train.csv', '2-Validate.csv')


# for option in [(0,1)]:
# 	for nf in [30]:
# 		for algo in ['SVR']:
# 			averageScore, sumTime = crossValidation(0, allFeatures = option[0], rankMethod = option[1], numFeatures = nf, algorithm = algo)

# -------------------------------------------------------------------------------------------------------
# options = [(0,	0,	'RBF',	2),
# (0,	2,	'RBF',	1),
# (1,	0,	'RBF',	16),
# (1,	2,	'RBF',	2),
# (2,	0,	'ML9',	3),
# (2,	2,	'RBF',	6),
# (3,	0,	'RBF',	7),
# (3,	2,	'RBF',	13),
# (4,	1,	'RBF',	18),
# (4,	3,	'RBF',	21),
# (5,	1,	'RBF',	2),
# (5,	2,	'RBF',	47),
# (6,	1,	'ML6',	4),
# (6,	3,	'ML6',	42),
# (7,	1,	'RBF',	2),	
# (7,	2,	'RBF',	48)]
options = [(0,	0,	'ELA',	1),(0,	1,	'ELA',	1),
(1,	0,	'ELA',	1),(1,	1,	'ELA',	1),
(2,	0,	'RID',	2),(2,	1,	'RID',	1),
(3,	0,	'RBF',	1),(3,	1,	'ELA',	15),
(4,	0,	'RBF',	1),(4,	1,	'RBF',	1),
(5,	0,	'ELA',	1),(5,	1,	'ELA',	1),
(6,	0,	'RBF',	2),(6,	1,	'RBF',	3),
(7,	0,	'SVL',	40),(7,	1,	'ELA',	1)]

for op in options:
	testModel('3-Test.csv','4-LargeTrain.csv', op[0], op[1], op[2], op[3],gridSearch = True)
	# testModel('New_2014-2015_Test_AllColsMissingBlank.csv','New_2006-2013_FilteredColsTargetMissingBlank.csv', op[0], op[1], op[2], op[3])

	# options = [(0,0,'MLP',1),(0,3,'RID',50),
# (1,1,'RID',5),(1,2,'LAS',47),
# (2,1,'SVL',1),(2,2,'RID',46),
# (3,0,'SVL',1),(3,3,'RID',46),
# (4,0,'RBF',1),(4,1,'RBF',1),(4,2,'RID',50),
# (5,0,'SVL',2),(5,3,'ML9',34),
# (6,0,'RBF',1),(6,3,'RID',21),
# (7,1,'MLP',1),(7,2,'ML8',44)]

# options = [(0,0,'ML1',1),(0,3,'RID',43),
# (1,1,'ML4',3),(1,2,'ELA',48),
# (2,1,'SVL',1),(2,3,'RID',34),
# (3,0,'RBF',1),(3,3,'RID',45),
# (4,0,'RBF',1),(4,2,'RID',45),
# (5,1,'MLP',2),(5,3,'ML9',46),
# (6,0,'RBF',1),(6,3,'RID',20),
# (7,1,'SVL',3),(7,2,'ML8',47)]

# options = [(0,0,'ML4',1),(0,3,'RID',43),
# (1,1,'ELA',8),(1,2,'RID',50),
# (2,0,'SVL',2),(2,3,'RID',38),
# (3,1,'SVL',1),(3,3,'RID',43),
# (4,1,'RID',18),(4,2,'RID',50),
# (5,0,'ML3',1),(5,3,'ML9',44),
# (6,1,'RID',35),(6,3,'RID',21),
# (7,0,'SVL',6),(7,3,'ML9',49)]
