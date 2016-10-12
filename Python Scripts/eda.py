import scipy.stats
import os
import re
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as co
import numpy as np

def spearmanAllFiles(folderName, filePrefix):
	fileList = []
	for root, dirs, files in os.walk('./'+folderName):    
	    for afile in files:
	    	fileList.append(afile)
	regex = re.compile("("+filePrefix+")"+".*\.(csv)")
	targetFiles = [m.group(0) for l in fileList for m in [regex.search(l)] if m]
	
	for afile in targetFiles:
		f = open('./'+folderName+'/'+afile,'rb')
		reader = csv.reader(f)
		header = next(reader)
		MINEResults = list(reader)
		MIC = []
		rho = []
		for row in MINEResults:
			MIC.append(row[2])
			rho.append(row[7])
		print ('%s - Spearman = %f, pvalue = %f')%(afile, scipy.stats.spearmanr(MIC,rho)[0], scipy.stats.spearmanr(MIC,rho)[1])

def spearmanTwoPhase():
	targetList = ['0808','0811','2704','2707','2713','2716','2718']
	for target in targetList:
		allYearValue = []
		oneYearValue = []

		f = open('./For Clustering/MINE-Indicator-2006TO2013-'+target+'.csv','rb')
		reader = csv.reader(f)
		header = next(reader)
		MINEResults = list(reader)
		indicatorDict = dict()
		for row in MINEResults:
			indicatorDict[row[1]] = row[2]
		f.close()

		f = open('./For Clustering/MINE-Indicator-2011-'+target+'.csv','rb')
		reader = csv.reader(f)
		header = next(reader)
		MINEResults = list(reader)
		for row in MINEResults:
			if indicatorDict.has_key(row[1][:row[1].rfind(' - ')]):
				allYearValue.append(indicatorDict[row[1][:row[1].rfind(' - ')]])
				oneYearValue.append(row[2])
		f.close()

		print target
		print len(allYearValue)
		print ('%s - Spearman = %f, pvalue = %f')%(target, scipy.stats.spearmanr(allYearValue,oneYearValue)[0], scipy.stats.spearmanr(allYearValue,oneYearValue)[1])

def plotMICRHOAllFiles(folderName, filePrefix):
	fileList = []
	for root, dirs, files in os.walk('./'+folderName):    
	    for afile in files:
	    	fileList.append(afile)
	regex = re.compile("("+filePrefix+")"+".*(-result\.arff)")
	targetFiles = [m.group(0) for l in fileList for m in [regex.search(l)] if m]

	for afile in targetFiles:
		f = open('./'+folderName+'/'+afile,'rb')
		reader = csv.reader(f)
		for i in range(14):
			header = next(reader)
		MINEResults = list(reader)
		clusters = dict()
		for row in MINEResults:
			(mic, rho, cluster) = (row[3], row[8], row[9])
			oldData = clusters.get(cluster,{'MIC':[],'RHO':[]})
			oldData['MIC'].append(mic)
			oldData['RHO'].append(rho)
			clusters[cluster] = oldData 

		colors = iter(cm.hsv(np.linspace(0, 1, len(clusters)+1)))
		keylist = clusters.keys()
		keylist.sort()
		plt.xlim((0,1))
		plt.ylim((-1,1))
		plotList = []
		for key in keylist:
			thisColor = next(colors)
			thisColor[3] = 0.5
			# plt.scatter(clusters[key]['MIC'],clusters[key]['RHO'],color=next(colors))
			plotList.append(plt.scatter(clusters[key]['MIC'],clusters[key]['RHO'],color=thisColor))
		plt.ylabel('Pearson\'s Correlation')
		plt.xlabel('MIC')
		plt.title(afile)
		plt.legend(tuple(plotList),
           tuple(['C'+str(i) for i in range(len(keylist))]),
           scatterpoints=1,
           # loc='lower left',
           loc='center right',
           # loc='upper center', bbox_to_anchor=(0.5, -0.05),
           ncol=1,
           fontsize=10)
		# plt.show()
		saveName = './'+folderName+'/'+afile[:-5]+'-RepairColor.png'
		plt.savefig(saveName)
		plt.close('all')
		# break

# spearmanTwoPhase()
# ======================================================================
# spearmanAllFiles('For Clustering','MINE-Indicator-2011-')
# spearmanAllFiles('For Clustering','MINE-Indicator-2006TO2013-')
# ======================================================================
plotMICRHOAllFiles('For Clustering','MINE-Indicator-2011-')
plotMICRHOAllFiles('For Clustering','MINE-Indicator-2006TO2013-')

