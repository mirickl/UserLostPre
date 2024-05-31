import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import OneHotEncoder
#%matplotlib inline


def divid_variable(df):
    type_list = ['int64','object']
    feature_str = df.select_dtypes(include=type_list)
    feature_num = df.select_dtypes(include='float64')
    return feature_str.columns,feature_num.columns

def calulate_p(df,futures):
    df_tmp = df[futures]

    cor = df_tmp.corr()  # 默认method='pearson'
    res  = []

    for i in range(len(futures)):
        for j in range(i+1,len(futures)):
            if cor.iloc[i][j] >= 0.6:
                #print(futures[i],futures[j])
                res.append((futures[i],futures[j]))
    return res

def calulate_k(df,furtures):
    # 定义函数计算Cramer's V系数
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def ont_hot_code(df):
    encodr = OneHotEncoder(sparse=False)
    ans = encodr.fit_transform(df)
    return ans


if __name__ == '__main__':
    df = pd.read_csv(r'D:\联通\移网流失率分析\四月流失预测_预览.csv', index_col=False, encoding='GB2312')
    # futures = ['DAYFEE', 'MONFEE', 'BALANCE', 'LEAVE_REAL_FEE', 'YUEZUFEE', 'YUYINFEE', 'DUANXINFEE', 'GPRSFEE', 'XLFEE','OTHERFEE']

    #print(cor_p)
    feature_str,feature_num = divid_variable(df)


    # 计算相关性
    print(feature_str)
    # print(df[feature_num])
    # cor_p = calulate_p(df, feature_num)
    # print(cor_p)
    # for future in feature_str






