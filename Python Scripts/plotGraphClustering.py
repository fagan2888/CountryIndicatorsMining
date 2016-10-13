import os
import re
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from operator import add

def distance(list1,list2):
	""" 
	This function finds Eucledian distance between two data points with the same number of attributes.

	Parameters
	----------
	list1, list2 : list of float
		Two lists which we want to know the distance between 

	Return Value
	----------
	If two lists have the same number of attributes, return the Eucledian distance. 
	Otherwise, return None.

	"""
	if len(list1) != len(list2):
		return None
	else:
		return np.sqrt(np.array([(list1[i]-list2[i])**2 for i in range(len(list1))]).sum())

def plot(XData,YData,XLabel,YLabel,title,saveName):
	""" 
	Generate and save a scatter plot.

	Parameters
	----------
	XData, YData : array_like, shape (n, )
		Input data

	XLabel,YLabel : string
		X axis label and Y axis label respectively

	title : string
		plot title

	saveName: string
		A filename or file path of the output plot (usually end with .png).

	Return Value
	----------
	None.

	"""
	plt.scatter(XData,YData)
	plt.ylabel(YLabel)
	plt.xlabel(XLabel)
	plt.title(title)
	plt.savefig(saveName)
	plt.close('all')

def plotPhase1():
	""" 
	Plot and save all scatter plots from iteration 1 into folders grouped by the clusters e.g. 0811N-1287.png.
	Moreover, duplicate the plot which is the nearest plot to the center of each cluster and save it e.g. Nearest-0811N-1071.png.
	Do this for all target indicators.

	** This function is hard coded. Please be careful when editing. **

	Parameters
	----------
	None

	Return Value
	----------
	None.

	"""
	targetIndex = ['0808','0811','1954','2704','2707','2713','2716','2718']
	for target in targetIndex:
		newpath = './For Clustering/'+target 
		if not os.path.exists(newpath):
			os.makedirs(newpath)

		filename = './For Clustering/MINE-Indicator-2011-'+target+'-result.arff'
		f = open(filename,'rb')
		while True:
			line = f.readline()
			if line.startswith('@attribute Cluster'):
				listCluster = line[line.index('{')+1:line.index('}')].split(',')
				break
			else:
				continue
		do = [os.makedirs('./For Clustering/'+target+'/'+cluster) for cluster in listCluster if not os.path.exists('./For Clustering/'+target+'/'+cluster)]
		f.close()

		pairs = np.genfromtxt(filename, delimiter=",", skip_header=14, autostrip=True, missing_values=np.nan, dtype=None)
		
		fname = './For Clustering/2011_2016-08-23-03-22-10.csv'
		with open(fname, 'r') as f:
			num_cols = len(f.readline().split())
			f.seek(0)
			data = np.genfromtxt(f, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, dtype=None)

		meanOf = dict()
		for row in pairs:
			
			cluster = row[9]
			XColumn = int(row[2][row[2].rfind('-')+1:-1].strip())+1
			YColumn = int(row[1][row[1].rfind('-')+1:-1].strip())+1
			
			XData = [country[XColumn] for country in data]
			YData = [country[YColumn] for country in data]
			# print XData
			# print YData
			plot(XData = XData, YData = YData, XLabel = row[2], YLabel = row[1], title = fname, saveName = './For Clustering/'+target+'/'+cluster+'/'+row[1][1:6]+'-'+row[2][1:5]+'.png')

			oldStat = meanOf.get(row[9], [0.0,0.0,0.0,0.0,0.0,0.0,0.0])
			newStat = [row[3],row[4],row[5],row[6],row[7],row[8],1.0]
			meanOf[row[9]] = map(add, oldStat, newStat)

		for key, value in meanOf.iteritems():
			mean = [x/value[-1] for x in value[:-1]]
			nearest = {'distance': None, 'row': None} # [MinDistance, NearestRow,XData,YData]
			for row in pairs:
				if row[9] == key:
					if (nearest['distance'] is None) or (distance(mean,[row[3],row[4],row[5],row[6],row[7],row[8]]) < nearest['distance']):
						nearest = {'distance': distance(mean,[row[3],row[4],row[5],row[6],row[7],row[8]]), 'row': row}

			XColumn = int(nearest['row'][2][nearest['row'][2].rfind('-')+1:-1].strip())+1
			YColumn = int(nearest['row'][1][nearest['row'][1].rfind('-')+1:-1].strip())+1
			XData = [country[XColumn] for country in data]
			YData = [country[YColumn] for country in data]
			plot(XData = XData, YData = YData, XLabel = nearest['row'][2], YLabel = nearest['row'][1], title = fname, saveName = './For Clustering/'+target+'/'+key+'/Nearest-'+nearest['row'][1][1:6]+'-'+nearest['row'][2][1:5]+'.png')

def plotPhase2():
	""" 
	Plot and save all scatter plots from iteration 2 into folders grouped by the clusters e.g. 0811N-1287.png.
	Moreover, duplicate the plot which is the nearest plot to the center of each cluster and save it e.g. Nearest-0811N-1071.png.
	Do this for all target indicators.

	** This function is hard coded. Please be careful when editing. **

	Parameters
	----------
	None

	Return Value
	----------
	None.

	"""
	targetIndex = ['0808','0811','0835','2704','2707','2713','2716','2718']
	for target in targetIndex:
		newpath = './For Clustering/'+target+'-2006TO2013' 
		if not os.path.exists(newpath):
			os.makedirs(newpath)

		filename = './For Clustering/MINE-Indicator-2006TO2013-'+target+'-result.arff'
		f = open(filename,'rb')
		while True:
			line = f.readline()
			if line.startswith('@attribute Cluster'):
				listCluster = line[line.index('{')+1:line.index('}')].split(',')
				break
			else:
				continue
		do = [os.makedirs('./For Clustering/'+target+'-2006TO2013/'+cluster) for cluster in listCluster if not os.path.exists('./For Clustering/'+target+'-2006TO2013/'+cluster)]
		f.close()

		pairs = np.genfromtxt(filename, delimiter=",", skip_header=14, autostrip=True, missing_values=np.nan, dtype=None)
		
		fname = './For Clustering/New_2006-2013_FilteredColsNotImputed.csv'
		with open(fname, 'r') as f:
			reader = csv.reader(f)
			header = next(reader)
			data = np.genfromtxt(fname, delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, dtype=None)
		f.close()
		
		meanOf = dict()
		for row in pairs:

			cluster = row[9]
			#Find Column
			XIndicatorKey = row[2][1:6]
			YIndicatorKey = row[1][1:6]
			# print XIndicatorKey, YIndicatorKey
			# print header
			XColumn = None
			YColumn = None
			for idx in range(len(header)):
				if header[idx].startswith(XIndicatorKey):
					XColumn = idx
				if header[idx].startswith(YIndicatorKey):
					YColumn = idx
				if XColumn is None or YColumn is None:
					continue
				else:
					break
			# print XColumn, YColumn
			XData = [country[XColumn] for country in data]
			YData = [country[YColumn] for country in data]
			# print XData
			# print YData
			plot(XData = XData, YData = YData, XLabel = row[2], YLabel = row[1], title = fname, saveName = './For Clustering/'+target+'-2006TO2013/'+cluster+'/'+row[1][1:6]+'-'+row[2][1:5]+'.png')

			oldStat = meanOf.get(row[9], [0.0,0.0,0.0,0.0,0.0,0.0,0.0])
			newStat = [row[3],row[4],row[5],row[6],row[7],row[8],1.0]
			meanOf[row[9]] = map(add, oldStat, newStat)

		for key, value in meanOf.iteritems():
			mean = [x/value[-1] for x in value[:-1]]
			nearest = {'distance': None, 'row': None} # [MinDistance, NearestRow,XData,YData]
			for row in pairs:
				if row[9] == key:
					if (nearest['distance'] is None) or (distance(mean,[row[3],row[4],row[5],row[6],row[7],row[8]]) < nearest['distance']):
						nearest = {'distance': distance(mean,[row[3],row[4],row[5],row[6],row[7],row[8]]), 'row': row}

			XIndicatorKey = nearest['row'][2][1:6]
			YIndicatorKey = nearest['row'][1][1:6]			
			XColumn = None
			YColumn = None
			for idx in range(len(header)):
				if header[idx].startswith(XIndicatorKey):
					XColumn = idx
				if header[idx].startswith(YIndicatorKey):
					YColumn = idx
				if XColumn is None or YColumn is None:
					continue
				else:
					break

			XData = [country[XColumn] for country in data]
			YData = [country[YColumn] for country in data]
			plot(XData = XData, YData = YData, XLabel = nearest['row'][2], YLabel = nearest['row'][1], title = fname, saveName = './For Clustering/'+target+'-2006TO2013/'+key+'/Nearest-'+nearest['row'][1][1:6]+'-'+nearest['row'][2][1:5]+'.png')

plotPhase2()
