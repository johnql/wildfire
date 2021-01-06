# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:56:03 2019

@author: johnw
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_1samp, f_oneway, pearsonr
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm
import researchpy as rp
from statsmodels.formula.api import ols

#data_xls= pd.read_excel('Canada 2008-2018 Forest Fires Jurisdiction.xlsx', index_col=None)
#data_xls.to_csv('cffj.csv', encoding='utf-8', index=False)

data=pd.read_csv('cffj.csv')
data=data.drop(['Protection zone','Response Category'], axis=1)
print(data.describe())
print(data.dtypes)
#, ordered=True
data['Fire Size Class'].astype('category')
data['Year'].astype('category')
data['Fire Size Class']=pd.Categorical(data['Fire Size Class'])
data['Year']=pd.Categorical(data['Year'])
print(data.dtypes)

#alb08=cffj_data.groupby(cffj_data['Jurisdiction']).[cffj_data['Year']==2008 & cffj_data['Fire Size Class']==1]
#print(alb08.head(5))
loca=data['Jurisdiction'].unique().tolist()
dataf_j_rs=loca
dataf = pd.DataFrame(columns=['Fire Size Class', 'Jurisdiction', 'Year', 'Number of Occurrences'])
dataf_j = pd.DataFrame(columns=['Jurisdiction', 'Year', 'Total Number of Occurrences'])
for loc in loca:
    for yea in range(2008,2019):
        tmpl=data[data['Jurisdiction']==loc]
        tmpl=tmpl.sort_values(by=['Fire Size Class','Year'])
        tmply=tmpl[tmpl['Year']==yea]
        tmplys=tmply.sum()
        tmplyn=tmplys['Number of Occurrences']
        dataf_j=dataf_j.append({'Jurisdiction' : loc, 'Year': yea, 'Total Number of Occurrences': tmplyn} , ignore_index=True)
        for cla in range(0,9):
            
            tmplyc=tmply[tmply['Fire Size Class']==cla].sum()
            tmplycn=tmplyc['Number of Occurrences']
#            print('Fire Size Class: ' + str(cla) + ',Jurisdiction:' + loc + ',Year:' + str(yea)+  ', Number of Occurrences:'+ str(tmplycn) + ', Total Number of Occurrences:'+ str(tmplyn))
            
            dataf=dataf.append({'Fire Size Class' : cla , 'Jurisdiction' : loc, 'Year': yea, 'Number of Occurrences': tmplycn} , ignore_index=True)
   
dataf['equal_or_higher_than_5'] = dataf['Fire Size Class'].apply(lambda x: 0 if x <= 4 else 1)
#####df['dose'].replace({1: 'placebo', 2: 'low', 3: 'high'}, inplace= True)
lt5c=[]
gt5c=[] #'Occurrences less than_5', 'Occurrences equal or larger than_5']
for loc in loca:
    for yea in range(2008,2019):
        #for l5 in range(0,2):
        tmpl=dataf[dataf['Jurisdiction']==loc]
        tmpl=tmpl.sort_values(by=['Year'])
        tmply=tmpl[tmpl['Year']==yea]
        tmplycl=tmply[tmply['equal_or_higher_than_5']==0].sum()
        tmplycln=tmplycl['Number of Occurrences']
        tmplycg=tmply[tmply['equal_or_higher_than_5']==1].sum()
        tmplycgn=tmplycg['Number of Occurrences']
#        print('less_than_5:'+ str(tmplycln)+ ', great_than_5:'+ str(tmplycgn))
        lt5c.append(tmplycln) #tmplycln=tmplycl['Number of Occurrences']
        gt5c.append(tmplycgn) #tmplycgn=tmplycg['Number of Occurrences']
dataf_j['Occurrences less than_5']=lt5c
dataf_j['Occurrences equal or larger than_5']=gt5c

occ_ltm=[]
occ_gtm=[]
occ_tot=[]
for loc in loca:
    tmpl=dataf_j[dataf_j['Jurisdiction']==loc]
    print(loc + ': ')
    print(tmpl.describe())
    occ_ltm.append(tmpl['Occurrences less than_5'].mean())
    occ_gtm.append(tmpl['Occurrences equal or larger than_5'].mean())
    occ_tot.append(tmpl['Total Number of Occurrences'].mean())
dataf_c = pd.DataFrame(
    {'Jurisdiction': loca,
     'Mean Occurrences less than_5': occ_ltm,
     'Mean Occurrences equal or larger than_5': occ_gtm
     
    })
    
#    'Mean Occurrences': occ_tot
#    tmpl['Occurrences less than_5'].descibe()
#    tmpl['Occurrences equal or larger than_5'].describe()
#albg08ex=albg08[albg08['Fire Size Class'].isin([5,6,7,8])].sum()
#albg0901sf=albg0901s['Number of Occurrences']

dataf_c.set_index(['Jurisdiction'], inplace = True)

fire_o_yb=[]
for yea in range(2008,2019):
    
        #for l5 in range(0,2):
    tmpy=dataf[dataf['Year']==yea]
    tmpy=tmpy.sort_values(by=['Year'])
    tmpy=tmpy.sum()
    
    tmpyn=tmpy['Number of Occurrences']
    fire_o_yb.append(tmpyn)


tmp_list = list(range(0,11))

print(tmp_list)
#print(stats.describe(fire_o_yb))

# solve for a and b
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

#z score deviation. chi-2 correlation z score. calculation summary conclusion
# pearson correlation test++++++++++++
    
dal=dataf_j[dataf_j['Jurisdiction']=='Alberta']['Occurrences equal or larger than_5']
dbc=dataf_j[dataf_j['Jurisdiction']=='British Columbia']['Occurrences equal or larger than_5']
dmt=dataf_j[dataf_j['Jurisdiction']=='Manitoba']['Occurrences equal or larger than_5']
dnp=dataf_j[dataf_j['Jurisdiction']=='National parks']['Occurrences equal or larger than_5']
dnb=dataf_j[dataf_j['Jurisdiction']=='New Brunswick']['Occurrences equal or larger than_5']
dnl=dataf_j[dataf_j['Jurisdiction']=='Newfoundland and Labrador']['Occurrences equal or larger than_5']
dnt=dataf_j[dataf_j['Jurisdiction']=='Northwest Territories']['Occurrences equal or larger than_5']
dns=dataf_j[dataf_j['Jurisdiction']=='Nova Scotia']['Occurrences equal or larger than_5']
dot=dataf_j[dataf_j['Jurisdiction']=='Ontario']['Occurrences equal or larger than_5']
dpe=dataf_j[dataf_j['Jurisdiction']=='Prince Edward Island']['Occurrences equal or larger than_5']
dqb=dataf_j[dataf_j['Jurisdiction']=='Quebec']['Occurrences equal or larger than_5']
dsk=dataf_j[dataf_j['Jurisdiction']=='Saskatchewan']['Occurrences equal or larger than_5']
dyk=dataf_j[dataf_j['Jurisdiction']=='Yukon']['Occurrences equal or larger than_5']

data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = pearsonr(dnt, dsk)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent between Saskatchewan and Northwest Territories with p:  ' + str(p))
else:
	print('Probably dependent'+ str(p))

#------------------------
#  correlation heatmap++++++++++++  
dataf_j=dataf_j.set_index(['Year'])
dataf_j_rs=pd.DataFrame() 
for loc in loca:
#   closing_df=pdr.get_data_yahoo(["AAPL","GOOG","MSFT","AMZN"],start,end)['Adj Close']
#   tech_rets = closing_df.pct_change()
    print(loc + 'rs ------')
  #  globals() [loc] = dataf_j[dataf_j['Jurisdiction']==loc]
    dataf_j_rs[loc]=dataf_j[dataf_j['Jurisdiction']==loc]['Occurrences equal or larger than_5'] 
    print(dataf_j[dataf_j['Jurisdiction']==loc]['Occurrences equal or larger than_5'])
  #  globals()[loc].info()
    
dataf_rets=dataf_j_rs.pct_change()

#dataf_j['Occurrences equal or larger than_5'].reshape(11,13)    
print(dataf_rets)
corr = dataf_rets.corr()
print(corr.describe())
plt.title('Correlation Heatmap among Canada Jurisdictions for Wild Fire > 100.1 he\n')
ttl = ax.title
ttl.set_position([.5, 1.05])
sns.heatmap(corr)    
#    
    
    
# ----------------------------    
    
    
# anova test +++++++++++++++++
#print(dataf_j[dataf_j['Jurisdiction']=='Alberta']['Occurrences equal or larger than_5'].head(2))
#dataf_j[dataf_j['Jurisdiction']=='Alberta']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='British Columbia']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Manitoba']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='National parks']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='New Brunswick']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Newfoundland and Labrador']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Northwest Territories']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Nova Scotia']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Ontario']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Prince Edward Island']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Quebec']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Saskatchewan']['Occurrences equal or larger than_5']
#dataf_j[dataf_j['Jurisdiction']=='Yukon']['Occurrences equal or larger than_5']
#------------------------
    

# bar chart for >5 occurance by location++++++++++++++++++++
#ax=dataf_c['Mean Occurrences equal or larger than_5'].plot(kind='barh', color='orange',stacked=False, title ="Past 10 Years Average Wild Fire based on Jurisdiction",  fontsize=20)
#leg_t=['> 100.1 he']
#plt.legend(leg_t)
#ax.set_xlabel('Average Occurrences', fontsize=12)
#ax.set_ylabel('Jurisdiction', fontsize=12)
#plt.show()
#----------------------------
# stacked bar chart for Jurisdiction comparison +++++++++++++
#dataf_c.drop['Mean Occurrences']
# 'Mean Occurrences less than_5','Mean Occurrences equal or larger than_5'
#ax=dataf_c.plot(kind='barh', stacked=True, title ="Past 10 Years Average Wild Fire based on Jurisdiction \n",  fontsize=12)
#leg_t=['< 100 he','> 100.1 he']
#plt.legend(leg_t)
#ax.set_xlabel('Average Occurrences', fontsize=10)
#ax.set_ylabel('Jurisdiction', fontsize=10)
#plt.savefig('stackedfire.png')
# ----------------------
    

# 10 years occurrence comparison among 3 top jurisdictions  +++++++++++++++++

#dataf_3c=dataf_j[dataf_j['Jurisdiction'].isin(['British Columbia','Alberta','Ontario'])]
#dataf_3c=dataf_3c.drop(['Occurrences less than_5'], axis=1)
#dataf_3c=dataf_3c.drop(['Occurrences equal or larger than_5'], axis=1)
#dataf_3c.set_index(['Year'], inplace = True)
#dataf_3c_al= dataf_3c[dataf_3c['Jurisdiction']=='Alberta']
#dataf_3c_al=dataf_3c_al['Total Number of Occurrences'].tolist()
#dataf_3c_bc= dataf_3c[dataf_3c['Jurisdiction']=='British Columbia']
#dataf_3c_bc=dataf_3c_bc['Total Number of Occurrences'].tolist()
#dataf_3c_on=dataf_3c[dataf_3c['Jurisdiction']=='Ontario']
#dataf_3c_on=dataf_3c_on['Total Number of Occurrences'].tolist()
#
#x = np.arange(11)  # the label locations
#width = 0.25  # the width of the bars
#fig, ax = plt.subplots()
#rects1 = ax.bar(x - width, dataf_3c_al, width, label='Alberta')
#rects2 = ax.bar(x, dataf_3c_bc, width, label='British Columbia')
#rects3 = ax.bar(x + width, dataf_3c_on, width, label='Ontario')
#
#ax.set_ylabel('Occurances')
#ax.set_title('Past 10 Years Wild Fire Occurrences History')
#ax.set_xticks(x)
#ax.set_xticklabels(range(2008, 2019))
#ax.legend()
# -------------------------------------------------------------
#bx=dataf_3c.plot(kind='bar',stacked=False, title ="Past 10 Years Wild Fire Occurrences History",  legend=True, fontsize=12)
# line chart for canada wide fire occurance++++++++++++++

#a, b = best_fit(tmp_list, fire_o_yb)
#print(stats.describe(fire_o_yb))
#print('mean of past decade wildfire occurrences in Canada: '+ str(np.mean(fire_o_yb))+ ', with Standard Deviation: ' +str(np.std(fire_o_yb)))
#yfit = [a + b * xi for xi in tmp_list]
#plt.plot(range(2008,2019),fire_o_yb, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
#plt.plot( range(2008,2019), yfit, marker='', color='olive', linewidth=2, linestyle='dashed', label="best fit line")
#plt.title('National Wide Wild Fire Occurrences in Canada')
#plt.ylabel('Occurances')
#
#plt.show()
#---------------------------------------------------
com_by=np.mean(fire_o_yb)
print(com_by)

ttest, pval = ttest_1samp(fire_o_yb, 7000)
print(pval)

print (dataf_j_rs['Alberta'][:])
stat, p = f_oneway(dataf_j_rs['Alberta'][:], dataf_j_rs['British Columbia'][:], dataf_j_rs['Manitoba'][:],dataf_j_rs['National parks'][:],dataf_j_rs['New Brunswick'][:],dataf_j_rs['Newfoundland and Labrador'][:],dataf_j_rs['Northwest Territories'][:],dataf_j_rs['Nova Scotia'][:],dataf_j_rs['Ontario'][:],dataf_j_rs['Prince Edward Island'][:],dataf_j_rs['Quebec'][:],dataf_j_rs['Saskatchewan'][:],dataf_j_rs['Yukon'][:])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

nat=[300,266,279,442,267,505,419,407,477,239,534]
nat=[23.07692308,20.46153846,21.46153846,34,20.53846154,38.84615385,32.23076923,31.30769231,36.69230769,18.38461538,41.07692308]
dataf_j_rs['National']=nat
from scipy.stats import ttest_ind
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]



def ttesting(X, Y):
    
    stat, p = ttest_ind(X,Y)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
    	print('Probably the same distribution')
    else:
    	print('Probably different distributions')
    return stat, p
tmp_t=[]
for loc in loca:
    print('ttesting compare ' + loc + ' with population with 0.05 confidence level' )
    stat, p = ttesting(dataf_j_rs[loc][:],dataf_j_rs['National'][:])
    tmp_t.append(p)
data_t=pd.DataFrame()
data_t['Jurisdiction']=loca
data_t['P value']=tmp_t

sta1= rp.summary_cont(dataf_j['Occurrences equal or larger than_5'].groupby(dataf_j['Jurisdiction']))
sta2=rp.summary_cont(dataf_j['Occurrences equal or larger than_5'].groupby(dataf_j.index)) 

import statsmodels.api as sm
from statsmodels.formula.api import ols
results = ols('dataf_j_rs ~ C(loca)', data=dataf_j_rs).fit()
print (results.summary())
# 1smp test
#Correct_result=0
#for i in range(len(fire_o_yb)):
#    ttest, pval = ttest_1samp(fire_o_yb[i], 6282)
#    print(fire_o_yb[i])
#    print(pval)
#    if pval  <  0.05: 
#        Correct_result +=1
#print('there are {:.2f} of location are significant'.format(Correct_result))
# best fit line solution ++++++++++++++++++
#a, b = best_fit(tmp_list, fire_o_yb)
#
#fig, ax = plt.subplots()
#plt.scatter(tmp_list, fire_o_yb)
#yfit = [a + b * xi for xi in tmp_list]
#plt.plot(tmp_list, yfit)
#plt.title('Best fit line for National Wide Wild Fire Occurrences in Canada')
#plt.ylabel('Occurances')
#ax.set_xticks(tmp_list)
#ax.set_xticklabels(range(2008, 2019))
#plt.show()
#--------------------

#for row in range(len(cffj_data)):
#    if cffj_data['Jurisdiction']==
#        dch.append(data['Close'][row]-data['Open'][row])
# PCA++++++++++++++
# Standardize the Data
