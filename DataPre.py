import pandas as pd


df = pd.read_csv(r'D:\联通\移网流失率分析\四月流失预测_预览.csv',index_col=False,encoding='GB2312')


de_tmp = df[['DAYFEE','MONFEE','BALANCE','LEAVE_REAL_FEE','YUEZUFEE','YUYINFEE','DUANXINFEE','GPRSFEE','XLFEE','OTHERFEE']]

print(de_tmp)
#进行相关性分析

