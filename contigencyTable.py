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
print("statistic                    df    value      prob")
print("chi-square                    {0:.0f}   {1:.4f}    {2:.4f}".format(df, qpstat, pvalue))

G2stat, pvalue, df, _ = chi2_contingency(obs, lambda_="log-likelihood")
print("Likelihood Ratio chi-square   {0:.0f}   {1:.4f}    {2:.4f}".format(df, G2stat, pvalue))

ssrc, ssr, ssc = 0,0,0
R = np.zeros(obs.shape[0])
C = np.zeros(obs.shape[1])
for i in range(obs.shape[0]):
    R[i] = i+1
for j in range(obs.shape[1]):
    C[j] = j+1
Rbar, Cbar = 0,0
for i in range(obs.shape[0]):
    for j in range(obs.shape[1]):
        Rbar += obs[i, j] * R[i]
        Cbar += obs[i, j] * C[j]
Rbar, Cbar = Rbar/n, Cbar/n
for i in range(obs.shape[0]):
    for j in range(obs.shape[1]):
        ssrc += obs[i, j] * ((R[i] - Rbar) * (C[j] - Cbar))
        ssr += obs[i, j] * ((R[i] - Rbar) * (R[i] - Rbar))
        ssc += obs[i, j] * ((C[j] - Cbar) * (C[j] - Cbar))
pearsonR = ssrc/math.sqrt(ssr*ssc)
QMH = (n-1)*pearsonR*pearsonR
pvalue = 1.0-stats.chi2.cdf(QMH,1)
print("Mantel-Haenszel Chi-Square    1    {0:.4f}    {1:.4f}".format(QMH, pvalue))

phiCoefficient = math.sqrt(qpstat/n)
contingencyCoefficient = math.sqrt(qpstat/(n+qpstat))
cramerV = phiCoefficient / math.sqrt(np.min(obs.shape)-1)

print("Phi Coefficient               -    {0:.4f}    ------".format(phiCoefficient))
print("Contingency Coefficient       -    {0:.4f}    ------".format(contingencyCoefficient))
print("Cramer's V                    -    {0:.4f}    ------".format(cramerV))
