                    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:35:37 2024

@author: sealso
"""
import pandas as pd   # primarily used for data wrangling, R eqv. data.frame and related operations
import numpy as np    # basic linear algebra like matrix product, inverse etc.
import seaborn as sns # for plotting, ggplot-wannabe
import scipy as sp    # offers many statisitcal functions
import math as math   # basic math functions
import spm1d as spm1d # some multivariate statistics

# pandas data reading
X = pd.read_csv('/Users/sealso/Research/Teaching/Data/wine.data.txt', sep=",", header=None)
X.shape

# There is one row per wine sample. The first column contains the cultivar of a wine sample (labelled 1, 2 or 3), 
# and the following thirteen columns contain the concentrations of the 13 different chemicals in that sample. The columns are separated by commas.

# assigning and setting column names using for loop

column_names = [None]*X.shape[1]
for i in range(X.shape[1]):
    if i == 0:
       column_names[i] = "cultivar" 
    else:
       column_names[i] = "chemical_" + str(i)
    
X.columns = column_names

# another approach to set up column names 

column_names = ["cultivar"] + ["chemical_"  + str(x) for x in range(1, X.shape[1])]
X.columns = column_names

# pairwise plot, ggally package in R does the same

sns_plot1 = sns.pairplot(X.loc[:, 'chemical_1':'chemical_5'])
sns_plot1.figure.savefig("/Users/sealso/Research/Teaching/Codes/Results/out1.png")

[sp.stats.pearsonr(X.loc[:, 'chemical_1'], X.loc[:, y]) for y in column_names[2:14]] # pearson r
[sp.stats.spearmanr(X.loc[:, 'chemical_1'], X.loc[:, y]) for y in column_names[2:14]] # spearman r

sns_plot2 = sns.pairplot(X.loc[:, 'cultivar':'chemical_5'], hue="cultivar", diag_kind="hist")
sns_plot2.figure.savefig("/Users/sealso/Research/Teaching/Codes/Results/out2.png")

sns_plot3 = sns.pairplot(X.loc[:, 'chemical_1':'chemical_5'], kind="hist")
sns_plot3.figure.savefig("/Users/sealso/Research/Teaching/Codes/Results/out3.png")

sns_plot4 = sns.pairplot(X.loc[:, 'chemical_1':'chemical_5'], kind="kde")
sns_plot4.figure.savefig("/Users/sealso/Research/Teaching/Codes/Results/out4.png")

sns_plot5 = sns.pairplot(X.loc[:, 'cultivar':'chemical_5'], hue="cultivar", kind="kde")
sns_plot5.figure.savefig("/Users/sealso/Research/Teaching/Codes/Results/out5.png")



# multivariate mean and covariance

cultivar1 = X.loc[X['cultivar'] == 1, 'chemical_1':'chemical_13']   
cultivar2 = X.loc[X['cultivar'] == 2, 'chemical_1':'chemical_13'] 
cultivar3 = X.loc[X['cultivar'] == 3, 'chemical_1':'chemical_13'] 

mean_1 = cultivar1.apply(np.mean, axis = 0)
mean_2 = cultivar2.apply(np.mean, axis = 0)
mean_3 = cultivar3.apply(np.mean, axis = 0)

cov_1 = np.cov(cultivar1.T)
cov_2 = np.cov(cultivar2.T)
cov_3 = np.cov(cultivar3.T)

np.diag(cov_1)
np.diag(cov_2)
np.diag(cov_3)

# example of Mahalanobis distance

sample_mean1 = np.array([10, 5, 4])
sample_mean2 = np.array([8, 3, 4])
rho = 0.9
sample_cov = np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
np.linalg.inv(sample_cov)

EQD = np.dot((sample_mean1 - sample_mean2), (sample_mean1 - sample_mean2))

MHD = np.dot(np.dot((sample_mean1 - sample_mean2), np.linalg.inv(sample_cov)), 
       (sample_mean1 - sample_mean2))
MHD

rho = 0.5
sample_cov = np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
np.linalg.inv(sample_cov)
MHD = np.dot(np.dot((sample_mean1 - sample_mean2), np.linalg.inv(sample_cov)), 
       (sample_mean1 - sample_mean2))
MHD

rho = 0.1
sample_cov = np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
np.linalg.inv(sample_cov)
MHD = np.dot(np.dot((sample_mean1 - sample_mean2), np.linalg.inv(sample_cov)), 
       (sample_mean1 - sample_mean2))
MHD

rho = 0.01
sample_cov = np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
np.linalg.inv(sample_cov)
MHD = np.dot(np.dot((sample_mean1 - sample_mean2), np.linalg.inv(sample_cov)), 
       (sample_mean1 - sample_mean2))
MHD



# wine example and MLE

data = cultivar1.loc[:,  'chemical_1':'chemical_3']
n,p = data.shape
mean_1 = data.apply(np.mean, axis = 0)
cov_1 =  np.cov(data.T)
Sigma = cov_1
Sigma_inv = np.linalg.inv(cov_1)


## when Sigma is known and mu is unknown

def MVN_mu(x):
  mu = x[range(p)]
  Log_L = 0
  for i in range(n):
   Log_L = Log_L - (np.dot(np.dot((data.iloc[i, :] - mu), Sigma_inv), 
          (data.iloc[i, :] - mu)))/2

  Log_L = Log_L - n/2*np.linalg.slogdet(Sigma)[1] 
  return(-Log_L)

MVN_mu(mean_1.values)

initial_guess = np.zeros(p)
MVN_mu(initial_guess)

result = sp.optimize.minimize(MVN_mu, initial_guess,  method='BFGS')
x = result.x
mu_MLE = x[range(p)]
mu_MLE - mean_1


## when mu is known and Sigma is unknown

mu = mean_1
def MVN_sigma(Sigma_vec):
    Sigma = np.zeros([p, p])
    Sigma[np.tril_indices(p, -1)] = Sigma_vec[range(p, len(Sigma_vec))]
    Sigma = (Sigma + Sigma.T)
    np.fill_diagonal(Sigma, Sigma_vec[range(p)])
    Sigma_inv = np.linalg.inv(Sigma)
    Log_L = 0
    for i in range(n):
     Log_L = Log_L - (np.dot(np.dot((data.iloc[i, :] - mu), Sigma_inv), 
          (data.iloc[i, :] - mu)))/2
  
    Log_L = Log_L - n/2*np.linalg.slogdet(Sigma)[1] 
    return(-Log_L)

cov_1_vec = np.append(np.diag(cov_1), cov_1[np.tril_indices(p, -1)])
MVN_sigma(cov_1_vec)

cov_0 = np.zeros([p, p])
np.fill_diagonal(cov_0, 1)
cov_0_vec = np.append(np.diag(cov_0), cov_0[np.tril_indices(p, -1)])

initial_guess = cov_0_vec
MVN_sigma(initial_guess)
#bnds = tuple((0,math.inf) for x in initial_guess[range(p)]) + tuple((-math.inf,math.inf) for x in initial_guess[range((p), len(initial_guess))])
#result_sigma = sp.optimize.minimize(MVN_sigma, initial_guess,  method='SLSQP', bounds=bnds)
result_sigma = sp.optimize.minimize(MVN_sigma, initial_guess,  method='L-BFGS-B', options={'maxiter': 500})


Sigma_MLE_vec = result_sigma.x
Sigma_MLE = np.zeros([p, p])
Sigma_MLE[np.tril_indices(p, -1)] = Sigma_MLE_vec[range(p, len(Sigma_MLE_vec))]
Sigma_MLE = (Sigma_MLE + Sigma_MLE.T)
np.fill_diagonal(Sigma_MLE, Sigma_MLE_vec[range(p)])
Sigma_MLE - cov_1
    

# Hotelling T2 

cultivar1 = X.loc[X['cultivar'] == 1, 'chemical_1':'chemical_13']   
cultivar2 = X.loc[X['cultivar'] == 2, 'chemical_1':'chemical_13'] 
cultivar3 = X.loc[X['cultivar'] == 3, 'chemical_1':'chemical_13'] 

mean_1 = cultivar1.apply(np.mean, axis = 0)
mean_2 = cultivar2.apply(np.mean, axis = 0)
mean_3 = cultivar3.apply(np.mean, axis = 0)

cov_1 = np.cov(cultivar1.T)
cov_2 = np.cov(cultivar2.T)
cov_3 = np.cov(cultivar3.T)

n1 = cultivar1.shape[0]
n2 = cultivar2.shape[0]
p = cultivar2.shape[1]

W1 = (n1 - 1)*cov_1
W2 = (n2 - 1)*cov_2
S_pl = 1/(n1 + n2 - 2)*(W1 + W2)
T2 = n1*n2/(n1 + n2)*np.dot((mean_1 - mean_2).T, np.dot(np.linalg.inv(S_pl), (mean_1-mean_2)))
T2_star = (n1 + n2 - p - 1)/(n1 + n2 - 2)/p*T2
F_val = sp.stats.f.sf(T2_star, p, (n1 + n2 - p - 1)) # sp.stats.f.ppf(0.95, p, (n1 + n2 - p - 1))

# same statistic using existing package 

T2    = spm1d.stats.hotellings2(cultivar1, cultivar2)
T2i   = T2.inference(0.05)
T2i
