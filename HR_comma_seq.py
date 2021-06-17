# coding=utf-8
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['font.size'] = 12


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def model(x, theta):
    return sigmoid(np.matmul(x, theta))


df = pd.read_csv(r'C:\Users\94584\Desktop\Math\statistics\data\HR_comma_sep.csv')
# pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 10)
# print(df)
# print(df.info())

print(df.describe())

# print(df.describe(include=['O']).T)

"""
# 画Box Plot
plt.figure(figsize=(14, 4), dpi=500)
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    plt.boxplot(df.iloc[:, i], showmeans=True)
    plt.xlabel(df.columns.values[i])
plt.show()

# 画直方图
plt.figure(figsize=(20, 4), dpi=500)
for i in range(5):
    data = df.iloc[:, i]
    K = int(1 + 4 * math.log(len(data), 10))
    # 划分一个比较"整"的区间
    bins = np.linspace(float('%.1g' % data.min()), float('%.1g' % data.max()), K + 1)
    ax = plt.subplot(1, 5, i + 1)
    ax.set_title(df.columns.values[i])
    plt.hist(data, bins=bins, density=True, edgecolor='black')
    plt.grid(alpha=0.5)
plt.show()

# 统计在职/离职数量, 画饼图
attr = [u'在职', u'离职']
leftornot = df.left.value_counts()

plt.pie(leftornot, labels=attr, autopct="%0.2f%%")
plt.legend()
plt.show()
"""

#用列联表分析last_evaluation与是否离职的关系
#按Q1, Q2, Q3作为界将last_evaluation分为评价为(优良中差)4类
le_q = np.zeros(3)
for i in range(3):
    le_q[i] = np.percentile(df.iloc[:, 1], 25+i*25)
print(le_q)
