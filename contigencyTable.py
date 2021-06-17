from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

obs = np.array([
    [69,28,68,51,6],
    [69,38,55,37,0],
    [90,47,94,94,16]
])
n = np.sum(obs)

qpstat, pvalue, df, _ = chi2_contingency(obs)
print("statistic                       df      value      prob")
print("chi-square                    {0:.4f}   {1:.4f}    {2:.4f}".format(df, qpstat, pvalue))

G2stat, pvalue, df, _ = chi2_contingency(obs, lambda_="log-likelihood")
print("Likelihood Ratio chi-square   {0:.4f}   {1:.4f}    {2:.4f}".format(df, G2stat, pvalue))

#QMHstat, pvalue, df, _ = chi2_contingency(obs, lambda_="log-likelihood")

phiCoefficient = math.sqrt(qpstat/n)
contingencyCoefficient = math.sqrt(qpstat/(n+qpstat))
cramerV = phiCoefficient / math.sqrt(np.min(obs.shape)-1)

print("Phi Coefficient               ------    {0:.4f}    ------".format(phiCoefficient))
print("Contingency Coefficient       ------    {0:.4f}    ------".format(contingencyCoefficient))
print("Cramer's V                    ------    {0:.4f}    ------".format(cramerV))
