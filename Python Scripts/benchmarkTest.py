import os
import re
import csv
import time
import json
import math
import numpy as np
import scipy as sp
from decimal import Decimal
from scipy.stats import chi2
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
from sklearn.kernel_ridge import *
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

targetIDList = [2704,2707,2713,2716,2718,808,811,835]

def benchmarkTest(filename, targetIdx, method):
	f = open(filename, 'rb')
	allData = json.load(f)
	# print allData["1"]
	error = []
	
	for key, val in allData.iteritems():
		if method == 'mean':
			err = meanBenchmark(val, targetIdx)
		elif method == 'combEW-AP':
			err = combEWBenchmark(val, targetIdx, option = 'AP')
		elif method == 'combEW-SP':
			err = combEWBenchmark(val, targetIdx, option = 'SP')
		elif method == 'combEW-BP':
			err = combEWBenchmark(val, targetIdx, option = 'BP')
		elif method == 'combIMSE-AP':
			err = combIMSEBenchmark(val, targetIdx, option = 'AP')
		elif method == 'combJMA-AP':
			err = combJMABenchmark(val, targetIdx, option = 'AP')
		elif method == 'combIMSE-SP':
			err = combIMSEBenchmark(val, targetIdx, option = 'SP')
		elif method == 'combJMA-SP':
			err = combJMABenchmark(val, targetIdx, option = 'SP')
		elif method == 'combIMSE-BP':
			err = combIMSEBenchmark(val, targetIdx, option = 'BP')
		elif method == 'combJMA-BP':
			err = combJMABenchmark(val, targetIdx, option = 'BP')
		elif method == 'PC':
			err = PCBenchmark(val, targetIdx)
		elif method == 'PC2':
			err = PC2Benchmark(val, targetIdx)
		elif method == 'SPC':
			err = SPCBenchmark(val, targetIdx)
		elif method == 'PC-SP':
			err = PCBenchmark(val, targetIdx, option = 'SP')
		elif method == 'PC2-SP':
			err = PC2Benchmark(val, targetIdx, option = 'SP')
		elif method == 'SPC-SP':
			err = SPCBenchmark(val, targetIdx, option = 'SP')
		elif method == 'PC-BP':
			err = PCBenchmark(val, targetIdx, option = 'BP')
		elif method == 'PC2-BP':
			err = PC2Benchmark(val, targetIdx, option = 'BP')
		elif method == 'SPC-BP':
			err = SPCBenchmark(val, targetIdx, option = 'BP')
		elif method == 'PC-BPG':
			err = PCBenchmark(val, targetIdx, option = 'BPG')
		elif method == 'PC2-BPG':
			err = PC2Benchmark(val, targetIdx, option = 'BPG')
		elif method == 'SPC-BPG':
			err = SPCBenchmark(val, targetIdx, option = 'BPG')
		elif method == 'RW':
			err = RWBenchmark(val, targetIdx)
		elif method == 'AR1':
			err = AR1Benchmark(val, targetIdx)
		elif method == 'noChange':
			err = noChangeBenchmark(val, targetIdx)	
		elif method == 'KRRLinear':
			err = KRRBenchmark(val, targetIdx, 'linear')
		elif method == 'KRRPoly2':
			err = KRRBenchmark(val, targetIdx, 'polynomial')
		elif method == 'KRRRBF':
			err = KRRBenchmark(val, targetIdx, 'rbf')	
		elif method == 'KRRLinear-SP':
			err = KRRBenchmark(val, targetIdx, 'linear', option = 'SP')
		elif method == 'KRRPoly2-SP':
			err = KRRBenchmark(val, targetIdx, 'polynomial', option = 'SP')
		elif method == 'KRRRBF-SP':
			err = KRRBenchmark(val, targetIdx, 'rbf', option = 'SP')	
		elif method == 'KRRLinear-BP':
			err = KRRBenchmark(val, targetIdx, 'linear', option = 'BP')
		elif method == 'KRRPoly2-BP':
			err = KRRBenchmark(val, targetIdx, 'polynomial', option = 'BP')
		elif method == 'KRRRBF-BP':
			err = KRRBenchmark(val, targetIdx, 'rbf', option = 'BP')	
		elif method == 'OurMIC':
			err = OurMethod(val, targetIdx, rankingMethod = 0)	
		elif method == 'OurSKB':
			err = OurMethod(val, targetIdx, rankingMethod = 1)
		elif method == 'OurMICBoot':
			err = OurMethod(val, targetIdx, rankingMethod = 0, bootstrapping = True)	
		elif method == 'OurSKBBoot':
			err = OurMethod(val, targetIdx, rankingMethod = 1, bootstrapping = True)					
		elif method == 'OurMICGS':
			err = OurMethod(val, targetIdx, rankingMethod = 0, gridSearch = True)	
		elif method == 'OurSKBGS':
			err = OurMethod(val, targetIdx, rankingMethod = 1, gridSearch = True)
		elif method == 'OurMICBootGS':
			err = OurMethod(val, targetIdx, rankingMethod = 0, bootstrapping = True, gridSearch = True)	
		elif method == 'OurSKBBootGS':
			err = OurMethod(val, targetIdx, rankingMethod = 1, bootstrapping = True, gridSearch = True)		
		if not (err is None):
			error.append(err)

	mean_score = np.array(error).mean()
	sd_score = np.array(error).std()
	return mean_score, sd_score 

def getTargetIndexInData(targetName, dataName, targetIdx):
	target = targetName[targetIdx]
	target = target[0:4]+target[5:]
	return dataName.index(target)

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

def meanBenchmark(countryObject, targetIdx):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		predicted = historyY.mean()
		return np.absolute(float(answer) - predicted)

def combEWBenchmark(countryObject, targetIdx, option = 'AP'):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
		if keepRows.sum() < 0.5*keepRows.shape[0]:
			return None

		if option == 'SP':
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = 50)
		elif option == 'BP':
			numFeaturesTest = [1,2,3,4,5,10,15,20,25,30,35,40,45,50]	
			bestParam = (None,None)
			for nfIdx in range(len(numFeaturesTest)):
				numFeatures = numFeaturesTest[nfIdx]
				XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
				
				y = targetY[keepRows].astype(np.float)
				y = y.reshape(-1,1)

				predicted = np.array(y.shape[0]*[0])
				for i in range(len(XIndex)):
					clf = linear_model.LinearRegression()
					X = data[keepRows,XIndex[i]].reshape(-1,1)
					predicted = predicted + cross_val_predict(clf, X, y, X.shape[0])
				cvscore = np.absolute(y-(1.0*predicted/len(XIndex))).mean()
				if bestParam[1] is None or cvscore < bestParam[1]:
					bestParam = (nfIdx, cvscore)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		else:
			XIndex = range(data.shape[1])

		predictEachModel = []
		for i in range(len(XIndex)):
			clf = linear_model.LinearRegression()
			
			X = data[keepRows,XIndex[i]].reshape(-1,1)
			y = targetY[keepRows].astype(np.float)
			y = y.reshape(-1,1)

			clf.fit(X, y)
			predictEachModel.append(clf.predict(data[-1,XIndex[i]])[0])
		predicted = np.array(predictEachModel).mean()
		return np.absolute(float(answer) - predicted)

def combIMSEBenchmark(countryObject, targetIdx, option = 'AP'):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
		if keepRows.sum() < 0.5*keepRows.shape[0]:
			return None

		if option == 'SP':
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = 50)
		elif option == 'BP':
			numFeaturesTest = [1,2,3,4,5,10,15,20,25,30,35,40,45,50]	
			bestParam = (None,None)
			for nfIdx in range(len(numFeaturesTest)):
				numFeatures = numFeaturesTest[nfIdx]
				XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
				
				y = targetY[keepRows].astype(np.float)
				y = y.reshape(-1,1)

				predicted = np.array(y.shape[0]*[0])
				sumWeight = 0.0
				for i in range(len(XIndex)):
					clf = linear_model.LinearRegression()
					X = data[keepRows,XIndex[i]].reshape(-1,1)
					crossValScoreList = cross_val_score(clf, X, y, scoring = 'mean_absolute_error', cv = y.shape[0])*(-1)
					mse = np.array([x*x for x in crossValScoreList]).mean()
					if mse == 0:
						mse = 0.0001
					predicted = predicted + (1.0/mse)*cross_val_predict(clf, X, y, X.shape[0])
					sumWeight += (1.0/mse)
				cvscore = np.absolute(y-(1.0*predicted/sumWeight)).mean()
				if bestParam[1] is None or cvscore < bestParam[1]:
					bestParam = (nfIdx, cvscore)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		else:
			XIndex = range(data.shape[1])

		predictEachModel = []
		weightEachModel = []
		for i in range(len(XIndex)):
			clf = linear_model.LinearRegression()
			
			X = data[keepRows,XIndex[i]].reshape(-1,1)
			y = targetY[keepRows].astype(np.float)
			y = y.reshape(-1,1)

			crossValScoreList = cross_val_score(clf, X, y, scoring = 'mean_absolute_error', cv = y.shape[0])*(-1)
			mse = np.array([x*x for x in crossValScoreList]).mean()

			if mse == 0:
				mse = 0.0001

			clf.fit(X, y)			
			predictEachModel.append(clf.predict(data[-1,XIndex[i]])[0])
			weightEachModel.append(1.0/mse)

		predicted = (np.array(predictEachModel)*np.array(weightEachModel)).sum() / np.array(weightEachModel).sum()
		return np.absolute(float(answer) - predicted)

def combJMABenchmark(countryObject, targetIdx, option = 'AP'):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
		if keepRows.sum() < 0.5*keepRows.shape[0]:
			return None

		if option == 'SP':
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = 50)
		elif option == 'BP':
			numFeaturesTest = [10,15,20,25,30,35,40,45,50]	
			bestParam = (None,None)
			for nfIdx in range(len(numFeaturesTest)):
				numFeatures = numFeaturesTest[nfIdx]
				XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
				
				X = data[keepRows,:]
				y = targetY[keepRows].astype(np.float)
				y = y.reshape(-1,1)

				error = []
				for i in range(y.shape[0]):
					thisTarget = y[i].reshape(1,-1)
					thisXTest = X[i,tuple(XIndex)].reshape(1,-1)

					thisYTrain = np.concatenate((y[0:i,:],y[i+1:y.shape[0],:]))
					thisXTrain = np.concatenate((X[0:i,tuple(XIndex)],X[i+1:X.shape[0],tuple(XIndex)]))

					newXForFeature = []
					newXTestForFeature = []
					for j in range(len(XIndex)):
						clf = linear_model.LinearRegression()

						newXForFeature.append(cross_val_predict(clf, thisXTrain[:, j].reshape(-1,1), thisYTrain, thisXTrain.shape[0]))

						clf.fit(thisXTrain[:, j].reshape(-1,1), thisYTrain)
						newXTestForFeature.append(clf.predict(thisXTest[:, j].reshape(-1,1)))

					newX = np.array(newXForFeature).transpose()[0]
					clf = linear_model.LinearRegression()
					# print newX
					clf.fit(newX,thisYTrain)

					error.append(np.absolute(thisTarget - clf.predict(np.array(newXTestForFeature).transpose()[0])[0]))

				cvscore = np.array(error).mean()
				if bestParam[1] is None or cvscore < bestParam[1]:
					bestParam = (nfIdx, cvscore)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		else:
			XIndex = range(data.shape[1])

		X = data[keepRows,:]
		y = targetY[keepRows].astype(np.float)
		y = y.reshape(-1,1)

		newX = []
		for j in range(X.shape[0]):
			XValidate = X[j]
			yValidate = y[j]

			XTrain = np.concatenate((X[0:j,:],X[j+1:,:]))
			yTrain = np.concatenate((y[0:j],y[j+1:]))
			# print yTrain
			thisX = []

			for i in range(len(XIndex)):
				clf = linear_model.LinearRegression()
				# print XTrain
				XTrainI = XTrain[:,XIndex[i]].reshape(-1,1)
				clf.fit(XTrainI, yTrain)
				thisX.append(clf.predict(np.array([XValidate[XIndex[i]]]))[0][0])
				# print clf.predict(np.array([XValidate[XIndex[i]]]))

			# print thisX
			newX.append(thisX)

		clf = linear_model.LinearRegression()
		# print np.array(newX).shape
		clf.fit(np.array(newX),y)

		thisX = []
		for i in range(len(XIndex)):
			clf2 = linear_model.LinearRegression()
			clf2.fit(X[:,XIndex[i]].reshape(-1,1), y)
			thisX.append(clf2.predict(np.array([X[-1,XIndex[i]]]))[0][0])
		predicted = clf.predict(np.array([thisX]))[0]

		return np.absolute(float(answer) - predicted)

def PCBenchmark(countryObject, targetIdx, option = 'AP'):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
		if keepRows.sum() < 0.5*keepRows.shape[0]:
			return None

		if option == 'SP':
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = 50)
		elif option == 'BP':
			numFeaturesTest = [10,15,20,25,30,35,40,45,50]	
			bestParam = (None,None)
			for nfIdx in range(len(numFeaturesTest)):
				numFeatures = numFeaturesTest[nfIdx]
				XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
				XAll = data[keepRows,:]
				X = XAll[:, tuple(XIndex)]
				y = targetY[keepRows].astype(np.float)
				y = y.reshape(-1,1)
				pca = PCA(n_components = 10) #Actual component = 7 according to the number of data points
				X = pca.fit_transform(X)
				clf = linear_model.LinearRegression()
				y = np.ravel(y)
				cvscore = (-1) * cross_val_score(clf, X, y, cv = y.shape[0]).mean()
				if bestParam[1] is None or cvscore < bestParam[1]:
					bestParam = (nfIdx, cvscore)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		elif option == 'BPG':
			numFeaturesTest = [10,15,20,25,30,35,40,45,50]	
			numPCUsed = [1,2,3,4,5,6,7]
			bestParam = (None,None,None)
			for nfIdx in range(len(numFeaturesTest)):
				for pcIdx in range(len(numPCUsed)):
					numFeatures = numFeaturesTest[nfIdx]
					XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
					XAll = data[keepRows,:]
					X = XAll[:, tuple(XIndex)]
					y = targetY[keepRows].astype(np.float)
					y = y.reshape(-1,1)
					pca = PCA(n_components = 10) #Actual component = 7 according to the number of data points
					X = pca.fit_transform(X)
					if X.shape[1] < numPCUsed[pcIdx]:
						continue
					clf = linear_model.LinearRegression()
					y = np.ravel(y)
					cvscore = (-1) * cross_val_score(clf, X[:,tuple(range(numPCUsed[pcIdx]))], y, cv = y.shape[0]).mean()
					if bestParam[1] is None or cvscore < bestParam[1]:
						bestParam = (nfIdx, cvscore, pcIdx)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		else:
			XIndex = range(data.shape[1])

		pca = PCA(n_components = 10) #Actual component = 7 according to the number of data points
		
		X = data[keepRows,:]
		X = X[:,tuple(XIndex)]
		y = targetY[keepRows].astype(np.float)
		y = y.reshape(-1,1)
		X = pca.fit_transform(X)
		clf = linear_model.LinearRegression()
		if option == 'BPG':
			X = X[:,tuple(range(numPCUsed[bestParam[2]]))]
		clf.fit(X, y)

		XTest = pca.transform(np.array([data[-1,tuple(XIndex)]]))
		if option == 'BPG':
			XTest = XTest[:,tuple(range(numPCUsed[bestParam[2]]))]
		predicted = clf.predict(XTest)
		return np.absolute(float(answer) - predicted)

def PC2Benchmark(countryObject, targetIdx, option = 'AP'):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
		if keepRows.sum() < 0.5*keepRows.shape[0]:
			return None

		if option == 'SP':
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = 50)
		elif option == 'BP':
			numFeaturesTest = [10,15,20,25,30,35,40,45,50]	
			bestParam = (None,None)
			for nfIdx in range(len(numFeaturesTest)):
				numFeatures = numFeaturesTest[nfIdx]
				XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
				XAll = data[keepRows,:]
				X = XAll[:, tuple(XIndex)]
				y = targetY[keepRows].astype(np.float)
				y = y.reshape(-1,1)
				pca = PCA(n_components = 10) #Actual component = 7 according to the number of data points
				X = pca.fit_transform(X)
				X = X[:,0:4]
				X = np.concatenate((X, X*X),axis=1)
				clf = linear_model.LinearRegression()
				y = np.ravel(y)
				cvscore = (-1) * cross_val_score(clf, X, y, cv = y.shape[0]).mean()
				if bestParam[1] is None or cvscore < bestParam[1]:
					bestParam = (nfIdx, cvscore)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		elif option == 'BPG':
			numFeaturesTest = [10,15,20,25,30,35,40,45,50]	
			numPCUsed = [1,2,3,4]
			bestParam = (None,None,None)
			for nfIdx in range(len(numFeaturesTest)):
				for pcIdx in range(len(numPCUsed)):
					numFeatures = numFeaturesTest[nfIdx]
					XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
					XAll = data[keepRows,:]
					X = XAll[:, tuple(XIndex)]
					y = targetY[keepRows].astype(np.float)
					y = y.reshape(-1,1)
					pca = PCA(n_components = 10) #Actual component = 7 according to the number of data points
					X = pca.fit_transform(X)
					if X.shape[1] < numPCUsed[pcIdx]:
						continue
					X = X[:,0:numPCUsed[pcIdx]]
					X = np.concatenate((X, X*X),axis=1)
					clf = linear_model.LinearRegression()
					y = np.ravel(y)
					cvscore = (-1) * cross_val_score(clf, X, y, cv = y.shape[0]).mean()
					if bestParam[1] is None or cvscore < bestParam[1]:
						bestParam = (nfIdx, cvscore, pcIdx)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		else:
			XIndex = range(data.shape[1])

		pca = PCA(n_components = 10)
		pcUsed = 4
		if option == 'BPG':
			pcUsed = numPCUsed[bestParam[2]]
		X = data[keepRows,:]
		X = X[:,tuple(XIndex)]
		y = targetY[keepRows].astype(np.float)
		y = y.reshape(-1,1)
		X = pca.fit_transform(X)
		X = X[:,0:pcUsed]
		X = np.concatenate((X, X*X),axis=1)

		clf = linear_model.LinearRegression()
		clf.fit(X, y)

		XTest = pca.transform(np.array([data[-1,tuple(XIndex)]]))
		XTest = XTest[:,0:pcUsed]
		XTest = np.concatenate((XTest, XTest*XTest),axis=1)
		predicted = clf.predict(XTest)
		return np.absolute(float(answer) - predicted)

def SPCBenchmark(countryObject, targetIdx, option = 'AP'):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
		if keepRows.sum() < 0.5*keepRows.shape[0]:
			return None

		if option == 'SP':
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = 50)
		elif option == 'BP':
			numFeaturesTest = [10,15,20,25,30,35,40,45,50]	
			bestParam = (None,None)
			for nfIdx in range(len(numFeaturesTest)):
				numFeatures = numFeaturesTest[nfIdx]
				XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
				XAll = data[keepRows,:]
				X = XAll[:, tuple(XIndex)]
				X = np.concatenate((X, X*X),axis=1)
				y = targetY[keepRows].astype(np.float)
				y = y.reshape(-1,1)
				pca = PCA(n_components = 10) #Actual component = 7 according to the number of data points
				X = pca.fit_transform(X)
				clf = linear_model.LinearRegression()
				y = np.ravel(y)
				cvscore = (-1) * cross_val_score(clf, X, y, cv = y.shape[0]).mean()
				if bestParam[1] is None or cvscore < bestParam[1]:
					bestParam = (nfIdx, cvscore)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		elif option == 'BPG':
			numFeaturesTest = [10,15,20,25,30,35,40,45,50]
			numPCUsed = [1,2,3,4,5,6,7]
			bestParam = (None,None,None)
			for nfIdx in range(len(numFeaturesTest)):
				for pcIdx in range(len(numPCUsed)):
					numFeatures = numFeaturesTest[nfIdx]
					XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
					XAll = data[keepRows,:]
					X = XAll[:, tuple(XIndex)]
					X = np.concatenate((X, X*X),axis=1)
					y = targetY[keepRows].astype(np.float)
					y = y.reshape(-1,1)
					pca = PCA(n_components = 10) #Actual component = 7 according to the number of data points
					X = pca.fit_transform(X)
					if X.shape[1] < numPCUsed[pcIdx]:
						continue
					clf = linear_model.LinearRegression()
					y = np.ravel(y)
					cvscore = (-1) * cross_val_score(clf, X[:,0:numPCUsed[pcIdx]], y, cv = y.shape[0]).mean()
					if bestParam[1] is None or cvscore < bestParam[1]:
						bestParam = (nfIdx, cvscore, pcIdx)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		else:
			XIndex = range(data.shape[1])

		pca = PCA(n_components = 10) #Actual component = 7 according to the number of data points
		
		X = data[keepRows,:]
		X = X[:,tuple(XIndex)]
		X = np.concatenate((X, X*X),axis=1)
		y = targetY[keepRows].astype(np.float)
		y = y.reshape(-1,1)
		X = pca.fit_transform(X)
		clf = linear_model.LinearRegression()
		if option == 'BPG':
			X = X[:,0:numPCUsed[bestParam[2]]]
		clf.fit(X, y)

		XTest = np.array([data[-1,tuple(XIndex)]])
		XTest = np.concatenate((XTest, XTest*XTest),axis=1)
		XTest = pca.transform(XTest)
		if option == 'BPG':
			XTest = XTest[:,0:numPCUsed[bestParam[2]]]
		predicted = clf.predict(XTest)
		return np.absolute(float(answer) - predicted)

def RWBenchmark(countryObject, targetIdx):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		change = np.array([historyY[i+1]-historyY[i] for i in range(len(historyY)-1)])
		if change.std() == 0:
			thisYearChange = change.mean()
		else:
			thisYearChange = np.random.normal(change.mean(),change.std(),1)		
		predicted = historyY[-1] + thisYearChange
		return np.absolute(float(answer) - predicted)

def AR1Benchmark(countryObject, targetIdx):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		ts = historyY
		x = ts[0:-1]
		y = ts[1:]
		p = sp.polyfit(x,y,1)
		beta = p[0]

		# Estimate c
		c = sp.mean(ts)*(1-beta)

		# Estimate the variance from the residuals of the OLS regression.
		yhat = sp.polyval(p,x)
		variance = sp.var(y-yhat)
		sigma = sp.sqrt(variance)
		
		# Predict
		if sigma == 0:
			noise = c
		else:
			noise = c + sp.random.normal(0, sigma, 1)
		predicted = beta*historyY[-1] + noise

	return np.absolute(float(answer) - predicted)

def noChangeBenchmark(countryObject, targetIdx):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		predicted = historyY[-1]
		return np.absolute(float(answer) - predicted)

def KRRBenchmark(countryObject, targetIdx, kernel, option = 'AP'):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
		if keepRows.sum() < 0.5*keepRows.shape[0]:
			return None

		if option == 'SP':
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = 50)
		elif option == 'BP':
			numFeaturesTest = [1,2,3,4,5,10,15,20,25,30,35,40,45,50]	
			bestParam = (None,None)
			for nfIdx in range(len(numFeaturesTest)):
				numFeatures = numFeaturesTest[nfIdx]
				XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeatures)
				XAll = data[keepRows,:]
				X = XAll[:, tuple(XIndex)]
				y = targetY[keepRows].astype(np.float)
				y = y.reshape(-1,1)
				clf = KernelRidge(kernel=kernel, degree=2, coef0=1)
				y = np.ravel(y)
				cvscore = (-1) * cross_val_score(clf, X, y, cv = y.shape[0]).mean()
				if bestParam[1] is None or cvscore < bestParam[1]:
					bestParam = (nfIdx, cvscore)
			XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = 1,numFeatures = numFeaturesTest[bestParam[0]])
		else:
			XIndex = range(data.shape[1])

		X = data[keepRows,:]
		X = X[:,tuple(XIndex)]
		y = targetY[keepRows].astype(np.float)
		y = y.reshape(-1,1)

		if len(XIndex) < 4:
			lambdaa = 1
		else:
			pca = PCA(n_components = 4) 			
			XPCA = pca.fit_transform(X)
			clf = linear_model.LinearRegression()
			cvpredict = cross_val_predict(clf, XPCA, y, cv = y.shape[0])
			ssres = y - cvpredict
			ssres = (ssres*ssres).sum()
			sstot = ((y - y.mean()) * (y - y.mean())).sum()
			if sstot == 0:
				sstot = 0.0001
			r2 = 1 - ssres/sstot
			N = data.shape[1]		

			bestParam = (None,None,None)
			for sIdx in range(5):
				for lIdx in range(5):
					lambdaa = getKernelParam(kernel, N, r2, sIdx, lIdx)
					clf = KernelRidge(alpha=lambdaa, kernel=kernel, degree=2, coef0=1)
					cvscore = cross_val_score(clf, X, y, cv = y.shape[0]).mean()
					if bestParam[2] is None or cvscore < bestParam[2]:
						bestParam = (sIdx, lIdx, cvscore)
			lambdaa = getKernelParam(kernel, N, r2, bestParam[0], bestParam[1])

		clf = KernelRidge(alpha=lambdaa, kernel=kernel, degree=2, coef0=1)
		clf.fit(X,y)
		predicted = clf.predict(data[-1,tuple(XIndex)].reshape(1,-1))
		return np.absolute(float(answer) - predicted)

def getKernelParam(kernel, N, r2, sIdx, lIdx):
	if kernel == 'linear':
		sigma0 = math.sqrt(N/2.0)
		sigma = [0.5*sigma0,sigma0,2*sigma0,4*sigma0,8*sigma0][sIdx]
		lambda0 = (1+N/(sigma*sigma))*(1-r2)/(r2)
		lambdaa = [lambda0/8.0,lambda0/4.0,lambda0/2.0,lambda0,2*lambda0][lIdx]
	elif kernel == 'polynomial':
		sigma0 = math.sqrt((N+2)/2.0)
		sigma = [0.5*sigma0,sigma0,2*sigma0,4*sigma0,8*sigma0][sIdx]
		lambda0 = (1+2*N/(sigma*sigma)+(N)*(N+2)/(sigma**4))*(1-r2)/(r2)
		lambdaa = [lambda0/8.0,lambda0/4.0,lambda0/2.0,lambda0,2*lambda0][lIdx]
	elif kernel == 'rbf':
		sigma0 = math.sqrt(chi2.ppf(0.95,N)) / math.pi
		sigma = [0.5*sigma0,sigma0,2*sigma0,4*sigma0,8*sigma0][sIdx]
		lambda0 = (1-r2)/(r2)
		lambdaa = [lambda0/8.0,lambda0/4.0,lambda0/2.0,lambda0,2*lambda0][lIdx]
	return lambdaa

def OurMethod(countryObject, targetIdx, rankingMethod, bootstrapping = False, gridSearch = False):
	target = np.array(countryObject['target'])
	data = np.array(countryObject['data'])

	historyY = data[:, getTargetIndexInData(countryObject['targetName'],countryObject['dataName'],targetIdx)]
	targetY = target[:,targetIdx]
	answer = targetY[-1]
	
	if answer == '':
		return None
	else:
		keepRows = np.array([True if x != '' else False for x in targetY[:-1]] + [False])
		if keepRows.sum() < 0.5*keepRows.shape[0]:
			return None
		XAll = data[keepRows,:]
		y = targetY[keepRows].astype(np.float)
		y = y.reshape(-1,1)

		numFeaturesTest = [1,2,3,4,5,10,15,20,25,30,35,40,45,50]	
		algos = ['SVL','RBF', 'RID', 'ELA', 'MLP','ML3','ML4','ML8','ML9']	
		
		# numFeaturesTest = [1,2,3,4,5]	
		# algos = ['SVL','RBF', 'RID', 'ELA']	

		bestParam = (None,None,None)
		for aIdx in range(len(algos)):
			for nfIdx in range(len(numFeaturesTest)):
				numFeatures = numFeaturesTest[nfIdx]
				algorithm = algos[aIdx]

				XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = rankingMethod,numFeatures = numFeatures)
				X = XAll[:, tuple(XIndex)]
				if len(XIndex) == 1:
					X = X.reshape(-1,1)
				y = y.reshape(-1,1)
				Xscaler = preprocessing.StandardScaler().fit(X)
				Xscaler.transform(X)
				Yscaler = preprocessing.StandardScaler().fit(y)

				estimator = getEstimator(algorithm,Yscaler,numFeatures)
				if estimator is None:
					return algorithm + ': Wrong Algorithm'

				y = np.ravel(y)
				cvscore = (-1) * cross_val_score(estimator, X, y, cv = y.shape[0]).mean()
				if bestParam[2] is None or cvscore < bestParam[2]:
					bestParam = (aIdx, nfIdx, cvscore)

		if not gridSearch:
			estimator = getEstimator(algos[bestParam[0]],Yscaler,numFeaturesTest[bestParam[1]])
		else:
			estimator = getEstimatorGridSearch(algos[bestParam[0]],Yscaler,numFeaturesTest[bestParam[1]])
		XIndex = getFeaturesIndex(countryObject['dataName'],'Feature Selection 2006-2013',targetIDList[targetIdx],rankingMethod = rankingMethod,numFeatures = numFeaturesTest[bestParam[1]])
		XTrain = data[keepRows,:]
		XTrain = XTrain[:,tuple(XIndex)]
		if len(XIndex) == 1:
			XTrain = XTrain.reshape(-1,1)
		Xscaler = preprocessing.StandardScaler().fit(XTrain)
		Xscaler.transform(XTrain)
		
		XTest = data[-1, tuple(XIndex)]
		XTest = XTest.reshape(1,-1)
		Xscaler.transform(XTest)

		if not bootstrapping:
			estimator.fit(XTrain, np.ravel(y))
			predicted = estimator.predict(XTest)
		else:
			predicted = bootstrappingPrediction(estimator, XTrain, y, XTest)

		return np.absolute(float(answer) - predicted)

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

def bootstrappingPrediction(estimator, Xready, y, XTestReady, B = 200, parametric = True):
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

# for alg in ['OurMICGS','OurSKBGS','OurMICBootGS','OurSKBBootGS']:
for alg in ['SPC-BPG']:
	print alg
	for i in range(8):
		print benchmarkTest('New_2006-2013_data.json',i,alg)

# print benchmarkTest('New_2006-2013_data.json',0,'PC')