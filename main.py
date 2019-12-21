# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:34:42 2019

@author: Koustubh
"""

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
from collections import defaultdict
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import K2Score
from pgmpy.estimators import ExhaustiveSearch
from pgmpy.estimators import HillClimbSearch
from timeit import default_timer as timer
from pgmpy.inference import VariableElimination

tab3Data = pd.read_csv('Table3.csv')
tab4Data = pd.read_csv('Table4.csv')
tab5Data = pd.read_csv('Table5.csv')
tab6Data = pd.read_csv('Table6.csv')
tab7Data = pd.read_csv('Table7.csv')
tab8Data = pd.read_csv('Table8.csv')
t3=tab3Data.iloc[0]
t4=tab4Data.iloc[0]
t5=tab5Data.iloc[0]
t6=tab6Data.iloc[0]
t7=tab7Data.iloc[0]
t8=tab8Data.iloc[0]
t3=t3.to_dict()
t4=t4.to_dict()
t5=t5.to_dict()
t6=t6.to_dict()
t7=t7.to_dict()
t8=t8.to_dict()

del t3['x1']
del t4['x2']
del t5['x3']
del t6['x4']
del t7['x5']
del t8['x6']

def getSeparateCPDs(tabData,tableNo):
	columnnames=tabData.columns.values.tolist()
	A_Given_B=[]
	for i in tabData.index:
		if(tabData.loc[i,columnnames[0]][-1]==tableNo and tabData.iloc[i,:-1][0][-1]==tableNo):
			newrow=[]
			newrow=list(tabData.iloc[i,1:])
			A_Given_B.append(newrow)
	return A_Given_B

X2_Given_X1=getSeparateCPDs(tab3Data,'2')
X4_Given_X1=getSeparateCPDs(tab3Data,'4')
X6_Given_X1=getSeparateCPDs(tab3Data,'6')
X3_Given_X2=getSeparateCPDs(tab4Data,'3')
X5_Given_X2=getSeparateCPDs(tab4Data,'5')
X2_Given_X3=getSeparateCPDs(tab5Data,'2')
X5_Given_X3=getSeparateCPDs(tab5Data,'5')
X6_Given_X3=getSeparateCPDs(tab5Data,'6')
X1_Given_X4=getSeparateCPDs(tab6Data,'1')
X2_Given_X4=getSeparateCPDs(tab6Data,'2')
X6_Given_X4=getSeparateCPDs(tab6Data,'6')
X2_Given_X5=getSeparateCPDs(tab7Data,'2')
X3_Given_X5=getSeparateCPDs(tab7Data,'3')
X1_Given_X6=getSeparateCPDs(tab8Data,'1')
X2_Given_X6=getSeparateCPDs(tab8Data,'2')
X3_Given_X6=getSeparateCPDs(tab8Data,'3')
X4_Given_X6=getSeparateCPDs(tab8Data,'4')


tab3Data=tab3Data.iloc[1:,]
tab3Data.loc[:,'x01'] *= t3.get('x01')
tab3Data.loc[:,'x11'] *= t3.get('x11')
tab3Data.loc[:,'x21'] *= t3.get('x21')
tab3Data.loc[:,'x31'] *= t3.get('x31')

tab4Data=tab4Data.iloc[1:,]
tab4Data.loc[:,'x02'] *= t4.get('x02')
tab4Data.loc[:,'x12'] *= t4.get('x12')
tab4Data.loc[:,'x22'] *= t4.get('x22')
tab4Data.loc[:,'x32'] *= t4.get('x32')
tab4Data.loc[:,'x42'] *= t4.get('x42')

tab5Data=tab5Data.iloc[1:,]
tab5Data.loc[:,'x03'] *= t5.get('x03')
tab5Data.loc[:,'x13'] *= t5.get('x13')
tab5Data.loc[:,'x23'] *= t5.get('x23')

tab6Data=tab6Data.iloc[1:,]
tab6Data.loc[:,'x04'] *= t6.get('x04')
tab6Data.loc[:,'x14'] *= t6.get('x14')
tab6Data.loc[:,'x24'] *= t6.get('x24')
tab6Data.loc[:,'x34'] *= t6.get('x34')

tab7Data=tab7Data.iloc[1:,]
tab7Data.loc[:,'x05'] *= t7.get('x05')
tab7Data.loc[:,'x15'] *= t7.get('x15')
tab7Data.loc[:,'x25'] *= t7.get('x25')
tab7Data.loc[:,'x35'] *= t7.get('x35')

tab8Data=tab8Data.iloc[1:,]
tab8Data.loc[:,'x06'] *= t8.get('x06')
tab8Data.loc[:,'x16'] *= t8.get('x16')
tab8Data.loc[:,'x26'] *= t8.get('x26')
tab8Data.loc[:,'x36'] *= t8.get('x36')
tab8Data.loc[:,'x46'] *= t8.get('x46')

t3.update(t4)
t3.update(t5)
t3.update(t6)
t3.update(t7)
t3.update(t8)


def findDependency(tabData):
		columnnames=tabData.columns.values.tolist()
		sumlist=list()
		for i in tabData.index:
			rowsum=0
			for j in range(1,len(columnnames)):
				rowsum+=abs(tabData.loc[i,columnnames[j]]-(t3.get(columnnames[j]) * t3.get(tabData.loc[i,columnnames[0]])))
			sumlist.append(rowsum)
		tabData['result']=sumlist
		
		dependencyDict={}
		for i in tabData.index:
			if(tabData.loc[i,columnnames[0]][-1]=='1'):
				if(dependencyDict.get("X%s-X1" %columnnames[0][-1])==None):
					dependencyDict["X%s-X1" %columnnames[0][-1]]=tabData.loc[i,'result']			   
				else:
					dependencyDict.update({"X%s-X1" %columnnames[0][-1]: dependencyDict.get("X%s-X1" %columnnames[0][-1])+tabData.loc[i,'result']})
			if(tabData.loc[i,columnnames[0]][-1]=='2'):
				if(dependencyDict.get("X%s-X2" %columnnames[0][-1])==None):
					dependencyDict["X%s-X2" %columnnames[0][-1]]=tabData.loc[i,'result']			   
				else:
					dependencyDict.update({"X%s-X2" %columnnames[0][-1]: dependencyDict.get("X%s-X2" %columnnames[0][-1])+tabData.loc[i,'result']})            
			if(tabData.loc[i,columnnames[0]][-1]=='3'):
				if(dependencyDict.get("X%s-X3" %columnnames[0][-1])==None):
					dependencyDict["X%s-X3" %columnnames[0][-1]]=tabData.loc[i,'result']			   
				else:
					dependencyDict.update({"X%s-X3" %columnnames[0][-1]: dependencyDict.get("X%s-X3" %columnnames[0][-1])+tabData.loc[i,'result']})
			if(tabData.loc[i,columnnames[0]][-1]=='4'):
				if(dependencyDict.get("X%s-X4" %columnnames[0][-1])==None):
					dependencyDict["X%s-X4" %columnnames[0][-1]]=tabData.loc[i,'result']			   
				else:
					dependencyDict.update({"X%s-X4" %columnnames[0][-1]: dependencyDict.get("X%s-X4" %columnnames[0][-1])+tabData.loc[i,'result']})
			if(tabData.loc[i,columnnames[0]][-1]=='5'):
				if(dependencyDict.get("X%s-X5" %columnnames[0][-1])==None):
					dependencyDict["X%s-X5" %columnnames[0][-1]]=tabData.loc[i,'result']			   
				else:
					dependencyDict.update({"X%s-X5" %columnnames[0][-1]: dependencyDict.get("X%s-X5" %columnnames[0][-1])+tabData.loc[i,'result']})
			if(tabData.loc[i,columnnames[0]][-1]=='6'):
				if(dependencyDict.get("X%s-X6" %columnnames[0][-1])==None):
					dependencyDict["X%s-X6" %columnnames[0][-1]]=tabData.loc[i,'result']			   
				else:
					dependencyDict.update({"X%s-X6" %columnnames[0][-1]: dependencyDict.get("X%s-X6" %columnnames[0][-1])+tabData.loc[i,'result']})
		return dependencyDict
dependencyDict={}
dependencyDict.update(findDependency(tab3Data))
dependencyDict.update(findDependency(tab4Data))
dependencyDict.update(findDependency(tab5Data))
dependencyDict.update(findDependency(tab6Data))
dependencyDict.update(findDependency(tab7Data))
dependencyDict.update(findDependency(tab8Data))

for key, value in sorted(dependencyDict.items(),reverse=True, key=lambda item: (item[1], item[0])):
    print ("%s: %s" % (key, value))


##################### bayesian Model ##################
G1 = BayesianModel()
G1.add_nodes_from(['x2','x3', 'x4','x5', 'x6'])

G1.add_edges_from([('x4', 'x6'), ('x6', 'x2'), ('x6', 'x3'), ('x3', 'x5')])
G1.active_trail_nodes('x5')

cpd_x14 = TabularCPD(variable='x4', variable_card=4,values=[[0.715],[0.105],[0.01],[0.17]])

cpd_x16 = TabularCPD(variable='x6', variable_card=5,values=X6_Given_X4, 
                    evidence=['x4'],
                   evidence_card=[4])
cpd_x12 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x13 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x15 = TabularCPD(variable='x5', variable_card=4,values=X5_Given_X3, 
                    evidence=['x3'],
                   evidence_card=[3])
G1.add_cpds(cpd_x14,cpd_x16,cpd_x13,cpd_x15,cpd_x12)

inference1 = BayesianModelSampling(G1)
sample1 = inference1.forward_sample(size=1000, return_type='dataframe')
scorer1  = K2Score(sample1)

print('Model 1 K2 Score: ' + str(scorer1.score(G1)))
#sample.to_csv('G1SampleData.csv')
############################# Part 3 ###################################
"""
M=G.to_markov_model()
print(M.nodes())"""

#######################Second Model#####################################
#G = BayesianModel()
"""
G.add_nodes_from(['x1','x2','x3', 'x4','x5', 'x6'])
print(len(G))
G.add_edges_from([('x3', 'x5'), ('x3', 'x6'), ('x5','x2'), ('x6', 'x4'), ('x4', 'x1')])"""
#G.active_trail_nodes('x5')
"""G = BayesianModel([('x2','x5')])
cpd_x3 = TabularCPD(variable='x2', variable_card=5,values=[[0.275],[0.32],[0.025],[0.17],[0.21]])

cpd_x2 = TabularCPD(variable='x5', variable_card=4,values=X5_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
"""
"""G = BayesianModel([('x2','x3')])
cpd_x3 = TabularCPD(variable='x2', variable_card=5,values=[[0.275],[0.32],[0.025],[0.17],[0.21]])

cpd_x2 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
"""
#######################Second Model#####################################
G2 = BayesianModel([('x1','x4'),('x1','x2'),('x2','x3'),('x3','x5'),('x4','x6')])
cpd_x23 = TabularCPD(variable='x1', variable_card=4,values=[[0.78],[0.015],[0.055],[0.15]])

cpd_x22 = TabularCPD(variable='x4', variable_card=4,values=X4_Given_X1, 
                    evidence=['x1'],
                   evidence_card=[4])
cpd_x25 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X1, 
                    evidence=['x1'],
                   evidence_card=[4])
cpd_x26 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x24 = TabularCPD(variable='x5', variable_card=4,values=X5_Given_X3, 
                    evidence=['x3'],
                   evidence_card=[3])
cpd_x21 = TabularCPD(variable='x6', variable_card=5,values=X6_Given_X4, 
                    evidence=['x4'],
                   evidence_card=[4])
G2.add_cpds(cpd_x23,cpd_x22,cpd_x25,cpd_x26,cpd_x24,cpd_x21)

inference2 = BayesianModelSampling(G2)
sample2 = inference2.forward_sample(size=1000, return_type='dataframe')
scorer2  = K2Score(sample2)

print('Model 2 K2 Score: ' + str(scorer2.score(G2)))
#sample.to_csv('G2SampleData.csv')

#######################Third Model#####################################
G3 = BayesianModel([('x6','x4'),('x6','x2'),('x6','x1'),('x2','x3'),('x2','x5')])
cpd_x33 = TabularCPD(variable='x6', variable_card=5,values=[[0.015],[0.32],[0.14],[0.315],[0.21]])

cpd_x32 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x35 = TabularCPD(variable='x1', variable_card=4,values=X1_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x36 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x34 = TabularCPD(variable='x4', variable_card=4,values=X4_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x31 = TabularCPD(variable='x5', variable_card=4,values=X5_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
G3.add_cpds(cpd_x33,cpd_x32,cpd_x35,cpd_x36,cpd_x34,cpd_x31)

inference3 = BayesianModelSampling(G3)
sample3 = inference3.forward_sample(size=1000, return_type='dataframe')
scorer3  = K2Score(sample3)

print('Model 3 K2 Score: ' + str(scorer3.score(G3)))
#sample.to_csv('G3SampleData.csv')

#######################Fourth Model#####################################
G4 = BayesianModel([('x6','x4'),('x6','x2'),('x2','x3'),('x2','x5'),('x4','x1')])
cpd_x43 = TabularCPD(variable='x6', variable_card=5,values=[[0.015],[0.32],[0.14],[0.315],[0.21]])

cpd_x42 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x45 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x46 = TabularCPD(variable='x5', variable_card=4,values=X5_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x44 = TabularCPD(variable='x4', variable_card=4,values=X4_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x41 = TabularCPD(variable='x1', variable_card=4,values=X1_Given_X4, 
                    evidence=['x4'],
                   evidence_card=[4])
G4.add_cpds(cpd_x43,cpd_x42,cpd_x45,cpd_x46,cpd_x44,cpd_x41)

inference4 = BayesianModelSampling(G4)
sample4 = inference4.forward_sample(size=1000, return_type='dataframe')
scorer4  = K2Score(sample4)
print('Model 4 K2 Score: ' + str(scorer4.score(G4)))
#sample.to_csv('G4SampleData.csv')

#######################Fifth Model#####################################
G5 = BayesianModel([('x6','x4'),('x6','x1'),('x1','x2'),('x2','x5'),('x2','x3')])
cpd_x53 = TabularCPD(variable='x6', variable_card=5,values=[[0.015],[0.32],[0.14],[0.315],[0.21]])

cpd_x52 = TabularCPD(variable='x1', variable_card=4,values=X1_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x55 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x56 = TabularCPD(variable='x5', variable_card=4,values=X5_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x54 = TabularCPD(variable='x4', variable_card=4,values=X4_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x51 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X1, 
                    evidence=['x1'],
                   evidence_card=[4])
G5.add_cpds(cpd_x53,cpd_x52,cpd_x55,cpd_x56,cpd_x54,cpd_x51)

inference5 = BayesianModelSampling(G5)
sample5 = inference5.forward_sample(size=1000, return_type='dataframe')
scorer5  = K2Score(sample5)
print('Model 5 K2 Score: ' + str(scorer5.score(G5)))
#sample.to_csv('G5SampleData.csv')

#######################Sixth Model#####################################
G6 = BayesianModel([('x4','x6'),('x6','x1'),('x1','x2'),('x2','x5'),('x2','x3')])
cpd_x63 = TabularCPD(variable='x4', variable_card=4,values=[[0.715],[0.105],[0.01],[0.17]])

cpd_x62 = TabularCPD(variable='x1', variable_card=4,values=X1_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x65 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x66 = TabularCPD(variable='x5', variable_card=4,values=X5_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x64 = TabularCPD(variable='x6', variable_card=5,values=X6_Given_X4, 
                    evidence=['x4'],
                   evidence_card=[4])
cpd_x61 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X1, 
                    evidence=['x1'],
                   evidence_card=[4])
G6.add_cpds(cpd_x63,cpd_x62,cpd_x65,cpd_x66,cpd_x64,cpd_x61)

inference6 = BayesianModelSampling(G6)
sample6 = inference6.forward_sample(size=1000, return_type='dataframe')
scorer6  = K2Score(sample6)
print('Model 6 K2 Score: ' + str(scorer6.score(G6)))
#sample.to_csv('G7SampleData.csv')

#######################Seventh Model#####################################
G7 = BayesianModel([('x1','x2'),('x1','x4'),('x2','x3'),('x3','x5')])
cpd_x73 = TabularCPD(variable='x1', variable_card=4,values=[[0.78],[0.015],[0.055],[0.15]])

cpd_x72 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X1, 
                    evidence=['x1'],
                   evidence_card=[4])
cpd_x75 = TabularCPD(variable='x4', variable_card=4,values=X4_Given_X1, 
                    evidence=['x1'],
                   evidence_card=[4])
cpd_x76 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x74 = TabularCPD(variable='x5', variable_card=4,values=X5_Given_X3, 
                    evidence=['x3'],
                   evidence_card=[3])

G7.add_cpds(cpd_x73,cpd_x72,cpd_x75,cpd_x76,cpd_x74)

inference7 = BayesianModelSampling(G7)
sample7 = inference7.forward_sample(size=1000, return_type='dataframe')
scorer7  = K2Score(sample7)
print('Model 7 K2 Score: ' + str(scorer7.score(G7)))
#sample.to_csv('G8SampleData.csv')

#######################Eighth Model#####################################
G8 = BayesianModel([('x6','x2'),('x6','x1'),('x2','x3')])
cpd_x83 = TabularCPD(variable='x6', variable_card=5,values=[[0.015],[0.32],[0.14],[0.315],[0.21]])

cpd_x82 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x85 = TabularCPD(variable='x1', variable_card=4,values=X1_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x86 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
G8.add_cpds(cpd_x83,cpd_x82,cpd_x85,cpd_x86)

inference8 = BayesianModelSampling(G8)
sample8 = inference8.forward_sample(size=1000, return_type='dataframe')
scorer8  = K2Score(sample8)
print('Model 8 K2 Score: ' + str(scorer8.score(G8)))
#sample.to_csv('G9SampleData.csv')

#######################Ninth Model#####################################
G9 = BayesianModel([('x4','x6'),('x6','x2'),('x6','x1'),('x2','x3')])
cpd_x93 = TabularCPD(variable='x4', variable_card=4,values=[[0.715],[0.105],[0.01],[0.17]])

cpd_x92 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x95 = TabularCPD(variable='x1', variable_card=4,values=X1_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x96 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
cpd_x94 = TabularCPD(variable='x6', variable_card=5,values=X6_Given_X4, 
                    evidence=['x4'],
                   evidence_card=[4])
G9.add_cpds(cpd_x93,cpd_x92,cpd_x95,cpd_x96,cpd_x94)

inference9 = BayesianModelSampling(G9)
sample9 = inference9.forward_sample(size=1000, return_type='dataframe')
scorer9  = K2Score(sample9)
print('Model 9 K2 Score: ' + str(scorer9.score(G9)))
#sample.to_csv('G9SampleData.csv')

#######################Tenth Model#####################################
G10 = BayesianModel([('x6','x2'),('x2','x3')])
cpd_x103 = TabularCPD(variable='x6', variable_card=5,values=[[0.015],[0.32],[0.14],[0.315],[0.21]])

cpd_x102 = TabularCPD(variable='x2', variable_card=5,values=X2_Given_X6, 
                    evidence=['x6'],
                   evidence_card=[5])
cpd_x106 = TabularCPD(variable='x3', variable_card=3,values=X3_Given_X2, 
                    evidence=['x2'],
                   evidence_card=[5])
G10.add_cpds(cpd_x103,cpd_x102,cpd_x106)

inference10 = BayesianModelSampling(G10)
sample10 = inference10.forward_sample(size=1000, return_type='dataframe')
scorer10  = K2Score(sample10)
print('Model 10 K2 Score: ' + str(scorer10.score(G10)))
#sample.to_csv('G10SampleData.csv')

###################TASK 3- Markov Conversion for 'th' Dataset and Time Comparison#########################
# Running any query to get the time
start=timer()
infr=VariableElimination(G10)
qer =infr.map_query(['x3','x2'],evidence={'x6':2})
end=timer()
print("Query results for 'th' Bayesian NW")
print(qer['x3'])
print(qer['x2'])
print("Execution Time for 'th' Bayesian NW",end-start)

markov_th=G10.to_markov_model()
start=timer()
inf=VariableElimination(markov_th)
qer =infr.map_query(['x3','x2'],evidence={'x6':2})
end=timer()
print("Query results for 'th' Markov NW")
print(qer['x3'])
print(qer['x2'])
print("Execution Time for 'th' Markov NW",end-start)

"""
#This below code I wrote to combine all the samples in one csv and compute K2 score on each of the models.
# This is not part of the main part of th code, i used it one time to get the results.
sampledata = pd.read_csv('G1SampleData.csv')
sampledata['x1']=''
sampledata = sampledata.reindex(sorted(sampledata.columns), axis=1)
#del sampledata['Unnamed: 0']
sampledata.to_csv('G1SampleData.csv',index=False)


frames = [pd.read_csv('G1SampleData.csv'),pd.read_csv('G2SampleData.csv'),pd.read_csv('G3SampleData.csv'),pd.read_csv('G4SampleData.csv'),pd.read_csv('G5SampleData.csv'),pd.read_csv('G6SampleData.csv'),pd.read_csv('G7SampleData.csv'),pd.read_csv('G8SampleData.csv'),pd.read_csv('G9SampleData.csv'),pd.read_csv('G10SampleData.csv')]
  
result = pd.concat(frames)
#sampledata.append(df1,ignore_index=True)
print(result.shape)

#result.to_csv('AllSampleData.csv',index=False)

result = result.sample(frac=1).reset_index(drop=True)
G = BayesianModel([('x6','x2'),('x6','x1'),('x2','x3')])
scorer  = K2Score(result[0:1000])
print('Model 1 K2 Score: ' + str(scorer.score(G)))
"""
################## Infering about Data using MAP_Queries###################################
G = BayesianModel([('x6','x2'),('x2','x3')]) 
G.add_node('x4')
G.add_node('x5')
G.add_node('x1')
G.fit(pd.read_csv('AllSampleData.csv'))
infr=VariableElimination(G)
qer =infr.map_query(['x3','x2','x4','x5','x1'],evidence={'x6':0})
print(qer['x5'])
print(qer['x4'])
print(qer['x3'])
print(qer['x2'])
print(qer['x1'])

################## Printing the Most Probable Occurance of 'th'###################################
Udf=pd.read_csv('AllSampleData.csv').groupby(pd.read_csv('AllSampleData.csv').columns.tolist(),as_index=False)
Uniquecount=Udf.size().reset_index().rename(columns={0:'u'})
maxval=Uniquecount.loc[Uniquecount['u'].idxmax()] #idxmin() is used to get low probability th
print("High Prob th")
print(maxval)
################################################## TASK 4#######################################

data=pd.read_csv('AND-Features.csv')
data1=data.loc[:,'f1':'f9']
 # getting the best model using hill climb search
hc = HillClimbSearch(data1, scoring_method=K2Score(data1))
best_model = hc.estimate()
print(best_model.edges())
G_AND_Best = BayesianModel([('f3', 'f4'), ('f3', 'f9'), ('f3', 'f8'), ('f5', 'f9'), ('f5', 'f3'), ('f9', 'f8'), ('f9', 'f7'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2'), ('f9', 'f4')])
G_AND_Best.fit(data1)
scorer  = K2Score(data1)
print('AND Part Best Model HC K2 Score: ' + str(scorer.score(G_AND_Best)))

########################## Model 2##############################
G_AND_1 = BayesianModel([('f3', 'f9'), ('f3', 'f8'), ('f5', 'f9'), ('f5', 'f3'), ('f9', 'f8'), ('f9', 'f7'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2'), ('f9', 'f4')])
G_AND_1.fit(data1)
scorer1  = K2Score(data1)
print('AND Part:- Model 2 K2 Score: ' + str(scorer1.score(G_AND_1)))

########################## Model 3##############################
G_AND_2 = BayesianModel([('f3', 'f8'), ('f5', 'f9'), ('f5', 'f3'), ('f9', 'f8'), ('f9', 'f7'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2'), ('f9', 'f4')])
G_AND_2.fit(data1)
scorer2  = K2Score(data1)
print('AND Part:- Model 3 K2 Score: ' + str(scorer2.score(G_AND_2)))

########################## Model 4##############################
G_AND_3 = BayesianModel([('f3', 'f8'), ('f5', 'f9'), ('f5', 'f3'), ('f9', 'f7'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2'), ('f9', 'f4')])
G_AND_3.fit(data1)
scorer3  = K2Score(data1)
print('AND Part:- Model 4 K2 Score: ' + str(scorer3.score(G_AND_3)))

########################## Model 5##############################
G_AND_4 = BayesianModel([('f4', 'f9'), ('f3', 'f8'), ('f5', 'f9'), ('f5', 'f3'), ('f9', 'f8'), ('f9', 'f7'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2')])
G_AND_4.fit(data1)
scorer4  = K2Score(data1)
print('AND Part:- Model 5 K2 Score: ' + str(scorer4.score(G_AND_4)))

####################### Comparison###################################
start=timer()
inf=VariableElimination(G_AND_Best)
qer =inf.query(['f4'],evidence={'f2':0})
end=timer()
print("Execution Time for AND Bayesian NW",end-start)
#print(qer['f4'])
############################## TASK 4 :-Markov Conversion################
markov_and=G_AND_Best.to_markov_model()
start=timer()
inf=VariableElimination(markov_and)
qer =inf.query(['f4'],evidence={'f2':0})
end=timer()
print("Execution Time for AND Markov NW",end-start)
#print(qer['f4'])

