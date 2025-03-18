# 加载模型训练过程中需要使用的模块
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # 集成学习中的随机森林
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import rcParams
import math
import time
import copy
config = {
          "font.family": 'serif',
          "font.size": 15,
          "mathtext.fontset": 'stix',
          "font.serif": ['Times New Roman']}
rcParams.update(config)


def rs(yp, yo):
    r = 1-np.sum((yp-yo)**2)/np.sum((yo-np.mean(yo))**2)
    return r


def rmse(yp, yo):
    z = np.linalg.norm(yp - yo, ord=2) / len(yp) ** 0.5
    return z


def mae(yp, yo):
    z = np.linalg.norm(yp-yo, ord=1)/len(yp)
    return z


def bias(yp, yo):
    z = np.sum(yp-yo)/len(yp)
    return z


save = 'model_tt-pic'
folder = os.path.split(os.path.realpath('model_tt-pic.py'))[0] + '\\' + save + '\\'
if not os.path.exists(folder):
    os.makedirs(folder)
# 读取当前数据集
df = pd.read_csv('D:\\ZHR_ML\\run_data-zhr03new.csv', sep=',', index_col=False)
ds = df.values
# ds = ds[ds[:, 7] < 4]
size = 0.20
ds_lab0 = ds[ds[:, 10] == 0]
ds_lab1 = ds[ds[:, 10] == 1]
ds_lab2 = ds[ds[:, 10] == 2]
ds_lab3 = ds[ds[:, 10] == 3]
ds_lab4 = ds[ds[:, 10] == 4]
ds_lab5 = ds[ds[:, 10] == 5]
ds_lab6 = ds[ds[:, 10] == 6]
Train_lab0, Test_lab0 = train_test_split(ds_lab0, test_size=size, random_state=6)
Train_lab1, Test_lab1 = train_test_split(ds_lab1, test_size=size, random_state=6)
Train_lab2, Test_lab2 = train_test_split(ds_lab2, test_size=size, random_state=6)
Train_lab3, Test_lab3 = train_test_split(ds_lab3, test_size=size, random_state=6)
Train_lab4, Test_lab4 = train_test_split(ds_lab4, test_size=size, random_state=6)
Train_lab5, Test_lab5 = train_test_split(ds_lab5, test_size=size, random_state=6)
Train_lab6, Test_lab6 = train_test_split(ds_lab6, test_size=size, random_state=6)
Train = np.concatenate((Train_lab0, Train_lab1,
                        Train_lab2, Train_lab3,
                        Train_lab4, Train_lab5), axis=0)
Train_lab = Train[Train[:, 10] <= 4]
Train_field = Train[Train[:, 10] > 4]
Test = np.concatenate((Test_lab0, Test_lab1,
                       Test_lab2, Test_lab3,
                       Test_lab4, Test_lab5), axis=0)
Test_lab = Test[Test[:, 10] <= 4]
Test_field = Test[Test[:, 10] > 4]
print(len(Train))
print(len(Test))
'''Training&Testing'''
np.random.seed(6)
np.random.shuffle(Train)
np.random.shuffle(Test)
X_Train = Train[:, 0:4]
Y_Train = Train[:, 7]
X_Test = Test[:, 0:4]
Y_Test = Test[:, 7]
X_Train = X_Train.astype('float')
Y_Train = Y_Train.astype('float')
X_Test = X_Test.astype('float')
Y_Test = Y_Test.astype('float')
# Training-lab
X_Train_lab = Train_lab[:, 0:4]
Y_Train_lab = Train_lab[:, 7]
X_Train_lab = X_Train_lab.astype('float')
Y_Train_lab = Y_Train_lab.astype('float')
# Training-field
X_Train_field = Train_field[:, 0:4]
Y_Train_field = Train_field[:, 7]
X_Train_field = X_Train_field.astype('float')
Y_Train_field = Y_Train_field.astype('float')
# Testing-lab
X_Test_lab = Test_lab[:, 0:4]
Y_Test_lab = Test_lab[:, 7]
X_Test_lab = X_Test_lab.astype('float')
Y_Test_lab = Y_Test_lab.astype('float')
# Testing-field
X_Test_field = Test_field[:, 0:4]
Y_Test_field = Test_field[:, 7]
X_Test_field = X_Test_field.astype('float')
Y_Test_field = Y_Test_field.astype('float')

'''RF模型'''
best_RF = RandomForestRegressor(n_estimators=450,
                                max_depth=10,
                                max_features=3,
                                random_state=6,
                                criterion='absolute_error',
                                n_jobs=-1)
best_RF.fit(X_Train, Y_Train)
# 保存模型
with open('D:\\ZHR_ML2\\' + save + '\\RF_model.pickle', 'wb') as f:
    pickle.dump(best_RF, f)
f.close()
# 载入模型验证数据结果
file = open('D:\\ZHR_ML2\\' + save + '\\RF_model.pickle', 'rb')
model_RF = pickle.load(file)
file.close()
'''GBDT模型'''
best_GBR = GradientBoostingRegressor(n_estimators=450,
                                     max_depth=9,
                                     max_features=3,
                                     subsample=0.3,
                                     learning_rate=0.01,
                                     random_state=6)
best_GBR.fit(X_Train, Y_Train)
# 保存模型
with open('D:\\ZHR_ML2\\' + save + '\\GBDT_model.pickle', 'wb') as f:
    pickle.dump(best_GBR, f)
f.close()
# 载入模型验证数据结果
file = open('D:\\ZHR_ML2\\' + save + '\\GBDT_model.pickle', 'rb')
model_GBDT = pickle.load(file)
file.close()
'''XG模型'''
best_XG = XGBRegressor(n_estimators=300,
                       max_depth=3,
                       subsample=0.6,
                       learning_rate=0.05,
                       random_state=6)
best_XG.fit(X_Train, Y_Train)
# 保存模型
with open('D:\\ZHR_ML2\\' + save + '\\XG_model.pickle', 'wb') as f:
    pickle.dump(best_XG, f)
f.close()
# 载入模型验证数据结果
file = open('D:\\ZHR_ML2\\' + save + '\\XG_model.pickle', 'rb')
model_XG = pickle.load(file)
file.close()
# '''DNN-2H模型'''
# model_DNN2H = load_model('D:\\ZHR_ML2\\DNN-2H_model.h5')
# '''DNN-3H模型'''
# model_DNN3H = load_model('D:\\ZHR_ML2\\DNN-3H_model.h5')

'''RF'''
'''(a)RF-Training训练集表现'''
pre_tr_lab = model_RF.predict(X_Train_lab)  # 模型预测
pre_tr_lab = pre_tr_lab.flatten()
pre_tr_field = model_RF.predict(X_Train_field)  # 模型预测
pre_tr_field = pre_tr_field.flatten()
r1_lab = 1-np.sum((pre_tr_lab-Y_Train_lab)**2)/np.sum((Y_Train_lab-np.mean(Y_Train_lab))**2)
rmse1_lab = rmse(pre_tr_lab, Y_Train_lab)
mae1_lab = mae(pre_tr_lab, Y_Train_lab)
r1_field = 1-np.sum((pre_tr_field-Y_Train_field)**2)/np.sum((Y_Train_field-np.mean(Y_Train_field))**2)
rmse1_field = rmse(pre_tr_field, Y_Train_field)
mae1_field = mae(pre_tr_field, Y_Train_field)
fig1, ax = plt.subplots(figsize=(6, 6), dpi=600)
stan = np.linspace(0, 10, 10)
s1_lab = plt.scatter(Y_Train_lab, pre_tr_lab, marker='o', s=75, c='mediumturquoise', edgecolors='k')
s1_field = plt.scatter(Y_Train_field, pre_tr_field, marker='X', s=100, c='salmon', edgecolors='k')
plt.legend(handles=[s1_lab, s1_field], labels=[
    'RF and laboratory dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_lab) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_lab)
    + ')',
    'RF and field dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_field) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_field)
    + ')'], loc='lower right', prop={'size': 13.5})
ax.set_facecolor("#FAFAEC")
ax.plot(stan, stan, linestyle='-', color='k', lw=1.5)
ax.plot(stan, stan*1.2, linestyle='--', color='#3639A2')
ax.plot(stan*1.2, stan, linestyle='--', color='#3639A2')
ax.plot(stan, stan*1.5, linestyle='-.', color='k')
ax.plot(stan*1.5, stan, linestyle='-.', color='k')
ax.text(.82, .95, '+20%',
        transform=ax.transAxes,
        color='#3639A2',
        zorder=4)
ax.text(.90, .84, '-20%',
        transform=ax.transAxes,
        color='#3639A2',
        zorder=4)
ax.text(.66, .95, '+50%',
        transform=ax.transAxes,
        color='k',
        zorder=4)
ax.text(.90, .56, '-50%',
        transform=ax.transAxes,
        color='k',
        zorder=4)
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# plt.grid(linestyle='--', linewidth=0.3, color='k', dashes=(10, 6))
plt.xlim((0, 10))
plt.ylim((0, 10))
plt.xlabel('Observed($y_s/H_s$)')
plt.ylabel('Predicted($y_s/H_s$)')
ax.text(.05, .92, '(a) Training results of the RF',
        transform=ax.transAxes,
        zorder=4)
plt.savefig('D:\\ZHR_ML2\\' + save + '\\(a)RF Model_Train.svg',
            dpi=600, bbox_inches="tight")  # 保存散点图
plt.show()  # 展示散点图
'''(b)RF-Testing训练集表现'''
pre_te_lab = model_RF.predict(X_Test_lab)  # 模型预测
pre_te_lab = pre_te_lab.flatten()
pre_te_field = model_RF.predict(X_Test_field)  # 模型预测
pre_te_field = pre_te_field.flatten()
r1_lab = 1-np.sum((pre_te_lab-Y_Test_lab)**2)/np.sum((Y_Test_lab-np.mean(Y_Test_lab))**2)
rmse1_lab = rmse(pre_te_lab, Y_Test_lab)
mae1_lab = mae(pre_te_lab, Y_Test_lab)
r1_field = 1-np.sum((pre_te_field-Y_Test_field)**2)/np.sum((Y_Test_field-np.mean(Y_Test_field))**2)
rmse1_field = rmse(pre_te_field, Y_Test_field)
mae1_field = mae(pre_te_field, Y_Test_field)
fig2, ax = plt.subplots(figsize=(6, 6), dpi=600)
stan = np.linspace(0, 10, 10)
s1_lab = plt.scatter(Y_Test_lab, pre_te_lab, marker='o', s=75, c='mediumturquoise', edgecolors='k')
s1_field = plt.scatter(Y_Test_field, pre_te_field, marker='X', s=100, c='salmon', edgecolors='k')
plt.legend(handles=[s1_lab, s1_field], labels=[
    'RF and laboratory dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_lab) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_lab)
    + ')',
    'RF and field dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_field) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_field)
    + ')'], loc='lower right', prop={'size': 13.5})
ax.set_facecolor("#FAFAEC")
ax.plot(stan, stan, linestyle='-', color='k', lw=1.5)
ax.plot(stan, stan*1.2, linestyle='--', color='#3639A2')
ax.plot(stan*1.2, stan, linestyle='--', color='#3639A2')
ax.plot(stan, stan*1.5, linestyle='-.', color='k')
ax.plot(stan*1.5, stan, linestyle='-.', color='k')
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim((0, 10))
plt.ylim((0, 10))
plt.xlabel('Observed($y_s/H_s$)')
plt.ylabel('Predicted($y_s/H_s$)')
ax.text(.05, .92, '(b) Testing results of the RF',
        transform=ax.transAxes,
        zorder=4)
plt.savefig('D:\\ZHR_ML2\\' + save + '\\(b)RF Model_Test.svg',
            dpi=600, bbox_inches="tight")  # 保存散点图
plt.show()  # 展示散点图
'''GBDT'''
'''(c)GBDT-Training训练集表现'''
pre_tr_lab = model_GBDT.predict(X_Train_lab)  # 模型预测
pre_tr_lab = pre_tr_lab.flatten()
pre_tr_field = model_GBDT.predict(X_Train_field)  # 模型预测
pre_tr_field = pre_tr_field.flatten()
r1_lab = 1-np.sum((pre_tr_lab-Y_Train_lab)**2)/np.sum((Y_Train_lab-np.mean(Y_Train_lab))**2)
rmse1_lab = rmse(pre_tr_lab, Y_Train_lab)
mae1_lab = mae(pre_tr_lab, Y_Train_lab)
r1_field = 1-np.sum((pre_tr_field-Y_Train_field)**2)/np.sum((Y_Train_field-np.mean(Y_Train_field))**2)
rmse1_field = rmse(pre_tr_field, Y_Train_field)
mae1_field = mae(pre_tr_field, Y_Train_field)
fig3, ax = plt.subplots(figsize=(6, 6), dpi=600)
stan = np.linspace(0, 10, 10)
s1_lab = plt.scatter(Y_Train_lab, pre_tr_lab, marker='o', s=75, c='mediumturquoise', edgecolors='k')
s1_field = plt.scatter(Y_Train_field, pre_tr_field, marker='X', s=100, c='salmon', edgecolors='k')
plt.legend(handles=[s1_lab, s1_field], labels=[
    'GBDT and laboratory dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_lab) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_lab)
    + ')',
    'GBDT and field dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_field) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_field)
    + ')'], loc='lower right', prop={'size': 13.5})
ax.set_facecolor("#FAFAEC")
ax.plot(stan, stan, linestyle='-', color='k', lw=1.5)
ax.plot(stan, stan*1.2, linestyle='--', color='#3639A2')
ax.plot(stan*1.2, stan, linestyle='--', color='#3639A2')
ax.plot(stan, stan*1.5, linestyle='-.', color='k')
ax.plot(stan*1.5, stan, linestyle='-.', color='k')
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim((0, 10))
plt.ylim((0, 10))
plt.xlabel('Observed($y_s/H_s$)')
plt.ylabel('Predicted($y_s/H_s$)')
ax.text(.05, .92, '(c) Training results of the GBDT',
        transform=ax.transAxes,
        zorder=4)
plt.savefig('D:\\ZHR_ML2\\' + save + '\\(c)GBDT Model_Train.svg',
            dpi=600, bbox_inches="tight")  # 保存散点图
plt.show()  # 展示散点图
'''(d)GBDT-Testing训练集表现'''
pre_te_lab = model_GBDT.predict(X_Test_lab)  # 模型预测
pre_te_lab = pre_te_lab.flatten()
pre_te_field = model_GBDT.predict(X_Test_field)  # 模型预测
pre_te_field = pre_te_field.flatten()
r1_lab = 1-np.sum((pre_te_lab-Y_Test_lab)**2)/np.sum((Y_Test_lab-np.mean(Y_Test_lab))**2)
rmse1_lab = rmse(pre_te_lab, Y_Test_lab)
mae1_lab = mae(pre_te_lab, Y_Test_lab)
r1_field = 1-np.sum((pre_te_field-Y_Test_field)**2)/np.sum((Y_Test_field-np.mean(Y_Test_field))**2)
rmse1_field = rmse(pre_te_field, Y_Test_field)
mae1_field = mae(pre_te_field, Y_Test_field)
fig4, ax = plt.subplots(figsize=(6, 6), dpi=600)
stan = np.linspace(0, 10, 10)
s1_lab = plt.scatter(Y_Test_lab, pre_te_lab, marker='o', s=75, c='mediumturquoise', edgecolors='k')
s1_field = plt.scatter(Y_Test_field, pre_te_field, marker='X', s=100, c='salmon', edgecolors='k')
plt.legend(handles=[s1_lab, s1_field], labels=[
    'GBDT and laboratory dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_lab) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_lab)
    + ')',
    'GBDT and field dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_field) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_field)
    + ')'], loc='lower right', prop={'size': 13.5})
ax.set_facecolor("#FAFAEC")
ax.plot(stan, stan, linestyle='-', color='k', lw=1.5)
ax.plot(stan, stan*1.2, linestyle='--', color='#3639A2')
ax.plot(stan*1.2, stan, linestyle='--', color='#3639A2')
ax.plot(stan, stan*1.5, linestyle='-.', color='k')
ax.plot(stan*1.5, stan, linestyle='-.', color='k')
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim((0, 10))
plt.ylim((0, 10))
plt.xlabel('Observed($y_s/H_s$)')
plt.ylabel('Predicted($y_s/H_s$)')
ax.text(.05, .92, '(d) Testing results of the GBDT',
        transform=ax.transAxes,
        zorder=4)
plt.savefig('D:\\ZHR_ML2\\' + save + '\\(d)GBDT Model_Test.svg',
            dpi=600, bbox_inches="tight")  # 保存散点图
plt.show()  # 展示散点图
'''XG'''
'''(e)XG-Training训练集表现'''
pre_tr_lab = model_XG.predict(X_Train_lab)  # 模型预测
pre_tr_lab = pre_tr_lab.flatten()
pre_tr_field = model_XG.predict(X_Train_field)  # 模型预测
pre_tr_field = pre_tr_field.flatten()
r1_lab = 1-np.sum((pre_tr_lab-Y_Train_lab)**2)/np.sum((Y_Train_lab-np.mean(Y_Train_lab))**2)
rmse1_lab = rmse(pre_tr_lab, Y_Train_lab)
mae1_lab = mae(pre_tr_lab, Y_Train_lab)
r1_field = 1-np.sum((pre_tr_field-Y_Train_field)**2)/np.sum((Y_Train_field-np.mean(Y_Train_field))**2)
rmse1_field = rmse(pre_tr_field, Y_Train_field)
mae1_field = mae(pre_tr_field, Y_Train_field)
fig5, ax = plt.subplots(figsize=(6, 6), dpi=600)
stan = np.linspace(0, 10, 10)
s1_lab = plt.scatter(Y_Train_lab, pre_tr_lab, marker='o', s=75, c='mediumturquoise', edgecolors='k')
s1_field = plt.scatter(Y_Train_field, pre_tr_field, marker='X', s=100, c='salmon', edgecolors='k')
plt.legend(handles=[s1_lab, s1_field], labels=[
    'XG and laboratory dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_lab) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_lab)
    + ')',
    'XG and field dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_field) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_field)
    + ')'], loc='lower right', prop={'size': 13.5})
ax.set_facecolor("#FAFAEC")
ax.plot(stan, stan, linestyle='-', color='k', lw=1.5)
ax.plot(stan, stan*1.2, linestyle='--', color='#3639A2')
ax.plot(stan*1.2, stan, linestyle='--', color='#3639A2')
ax.plot(stan, stan*1.5, linestyle='-.', color='k')
ax.plot(stan*1.5, stan, linestyle='-.', color='k')
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim((0, 10))
plt.ylim((0, 10))
plt.xlabel('Observed($y_s/H_s$)')
plt.ylabel('Predicted($y_s/H_s$)')
ax.text(.05, .92, '(e) Training results of the XG',
        transform=ax.transAxes,
        zorder=4)
plt.savefig('D:\\ZHR_ML2\\' + save + '\\(e)XG Model_Train.svg',
            dpi=600, bbox_inches="tight")  # 保存散点图
plt.show()  # 展示散点图
'''(f)XG-Testing训练集表现'''
pre_te_lab = model_XG.predict(X_Test_lab)  # 模型预测
pre_te_lab = pre_te_lab.flatten()
pre_te_field = model_XG.predict(X_Test_field)  # 模型预测
pre_te_field = pre_te_field.flatten()
r1_lab = 1-np.sum((pre_te_lab-Y_Test_lab)**2)/np.sum((Y_Test_lab-np.mean(Y_Test_lab))**2)
rmse1_lab = rmse(pre_te_lab, Y_Test_lab)
mae1_lab = mae(pre_te_lab, Y_Test_lab)
r1_field = 1-np.sum((pre_te_field-Y_Test_field)**2)/np.sum((Y_Test_field-np.mean(Y_Test_field))**2)
rmse1_field = rmse(pre_te_field, Y_Test_field)
mae1_field = mae(pre_te_field, Y_Test_field)
fig6, ax = plt.subplots(figsize=(6, 6), dpi=600)
stan = np.linspace(0, 10, 10)
s1_lab = plt.scatter(Y_Test_lab, pre_te_lab, marker='o', s=75, c='mediumturquoise', edgecolors='k')
s1_field = plt.scatter(Y_Test_field, pre_te_field, marker='X', s=100, c='salmon', edgecolors='k')
plt.legend(handles=[s1_lab, s1_field], labels=[
    'XG and laboratory dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_lab) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_lab)
    + ')',
    'XG and field dataset' + '\n'
    + '($R^2$'+'={:.3}'.format(r1_field) + ', '
    + '$MAE$'+'={:.3}'.format(mae1_field)
    + ')'], loc='lower right', prop={'size': 13.5})
ax.set_facecolor("#FAFAEC")
ax.plot(stan, stan, linestyle='-', color='k', lw=1.5)
ax.plot(stan, stan*1.2, linestyle='--', color='#3639A2')
ax.plot(stan*1.2, stan, linestyle='--', color='#3639A2')
ax.plot(stan, stan*1.5, linestyle='-.', color='k')
ax.plot(stan*1.5, stan, linestyle='-.', color='k')
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim((0, 10))
plt.ylim((0, 10))
plt.xlabel('Observed($y_s/H_s$)')
plt.ylabel('Predicted($y_s/H_s$)')
ax.text(.05, .92, '(f) Testing results of the XG',
        transform=ax.transAxes,
        zorder=4)
plt.savefig('D:\\ZHR_ML2\\' + save + '\\(f)XG Model_Test.svg',
            dpi=600, bbox_inches="tight")  # 保存散点图
plt.show()  # 展示散点图



'''DNN数据预处理'''
# # 计算当前训练集的归一化参数值
# mean_x_tr = X_Train.mean(axis=0)  # 按列对输入变量计算平均值
# std_x_tr = X_Train.std(axis=0)  # 按列对输入变量计算标准差
# # 归一化
# X_Train -= mean_x_tr  # 将训练样本减去平均值
# X_Train /= std_x_tr  # 再除以标准差
# X_Test -= mean_x_tr
# X_Test /= std_x_tr
# '''DNN-2H模型'''
# pre_tr = model_DNN2H.predict(X_Train)  # 模型预测
# pre_tr = pre_tr.flatten()
# pre_te = model_DNN2H.predict(X_Test)  # 模型预测
# pre_te = pre_te.flatten()
# r1 = 1-np.sum((pre_tr-Y_Train)**2)/np.sum((Y_Train-np.mean(Y_Train))**2)
# r2 = 1-np.sum((pre_te-Y_Test)**2)/np.sum((Y_Test-np.mean(Y_Test))**2)
# # 绘制预测值于实际值的散点图（展示训练效果）
# fig4, ax = plt.subplots(figsize=(5, 5), dpi=600)
# stan = np.linspace(0, 5, 10)
# s1 = plt.scatter(Y_Train, pre_tr, marker='o', c='mediumturquoise', edgecolors='k')
# s2 = plt.scatter(Y_Test, pre_te, marker='o', c='salmon', edgecolors='k')
# plt.legend(handles=[s1, s2], labels=['S3-DNN2H and' + '\n' + 'training dataset($R^2$'+'={:.3}'.format(r1) + ')',
#                                      'S3-DNN2H and' + '\n' + 'testing dataset($R^2$'+'={:.3}'.format(r2) + ')'],
#            loc='lower right', prop={'size': 11.5})
# ax.set_facecolor("#FAFAEC")
# ax.plot(stan, stan, linestyle='-', color='k', lw=1.0)
# ax.plot(stan, stan*1.2, linestyle='--', color='#3639A2')
# ax.plot(stan*1.2, stan, linestyle='--', color='#3639A2')
# ax.text(.50, .85, '+20% line',
#         transform=ax.transAxes,
#         zorder=4)
# ax.text(.80, .58, '-20% line',
#         transform=ax.transAxes,
#         zorder=4)
# x_major_locator = MultipleLocator(0.5)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# # plt.grid(linestyle='--', linewidth=0.35, color='k', dashes=(10, 6))
# plt.xlim((0, 4))
# plt.ylim((0, 4))
# plt.xlabel('Observed($y_s/H_s$)')
# plt.ylabel('Predicted($y_s/H_s$)')
# ax.text(.05, .92, '(d) DNN2H model',
#         transform=ax.transAxes,
#         zorder=4)
# plt.savefig('D:\\ZHR_ML2\\' + save + '\\(d)DNN2H Model.svg',
#             dpi=600, bbox_inches="tight")  # 保存散点图
# plt.show()  # 展示散点图
# '''DNN-3H模型'''
# pre_tr = model_DNN3H.predict(X_Train)  # 模型预测
# pre_tr = pre_tr.flatten()
# pre_te = model_DNN3H.predict(X_Test)  # 模型预测
# pre_te = pre_te.flatten()
# r1 = 1-np.sum((pre_tr-Y_Train)**2)/np.sum((Y_Train-np.mean(Y_Train))**2)
# r2 = 1-np.sum((pre_te-Y_Test)**2)/np.sum((Y_Test-np.mean(Y_Test))**2)
# # 绘制预测值于实际值的散点图（展示训练效果）
# fig5, ax = plt.subplots(figsize=(5, 5), dpi=600)
# stan = np.linspace(0, 5, 10)
# s1 = plt.scatter(Y_Train, pre_tr, marker='o', c='mediumturquoise', edgecolors='k')
# s2 = plt.scatter(Y_Test, pre_te, marker='o', c='salmon', edgecolors='k')
# plt.legend(handles=[s1, s2], labels=['S3-DNN3H and' + '\n' + 'training dataset($R^2$'+'={:.3}'.format(r1) + ')',
#                                      'S3-DNN3H and' + '\n' + 'testing dataset($R^2$'+'={:.3}'.format(r2) + ')'],
#            loc='lower right', prop={'size': 11.5})
# ax.set_facecolor("#FAFAEC")
# ax.plot(stan, stan, linestyle='-', color='k', lw=1.0)
# ax.plot(stan, stan*1.2, linestyle='--', color='#3639A2')
# ax.plot(stan*1.2, stan, linestyle='--', color='#3639A2')
# ax.text(.50, .85, '+20% line',
#         transform=ax.transAxes,
#         zorder=4)
# ax.text(.80, .58, '-20% line',
#         transform=ax.transAxes,
#         zorder=4)
# x_major_locator = MultipleLocator(0.5)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# # plt.grid(linestyle='--', linewidth=0.35, color='k', dashes=(10, 6))
# plt.xlim((0, 4))
# plt.ylim((0, 4))
# plt.xlabel('Observed($y_s/H_s$)')
# plt.ylabel('Predicted($y_s/H_s$)')
# ax.text(.05, .92, '(e) DNN3H model',
#         transform=ax.transAxes,
#         zorder=4)
# plt.savefig('D:\\ZHR_ML2\\' + save + '\\(e)DNN3H Model.svg',
#             dpi=600, bbox_inches="tight")  # 保存散点图
# plt.show()  # 展示散点图


# # 读取pic数据集
# df = pd.read_csv('D:\\ZHR_ML\\run_data-zhr03pic.csv', sep=',', index_col=False)
# ds = df.values
# ds = ds[ds[:, 7] < 4]
# size = 0.20
# ds_lab0 = ds[ds[:, 10] == 0]
# ds_lab1 = ds[ds[:, 10] == 1]
# ds_lab2 = ds[ds[:, 10] == 2]
# ds_lab3 = ds[ds[:, 10] == 3]
# ds_lab4 = ds[ds[:, 10] == 4]
# ds_lab5 = ds[ds[:, 10] == 5]
# ds_lab6 = ds[ds[:, 10] == 6]
# Train_lab0, Test_lab0 = train_test_split(ds_lab0, test_size=size, random_state=6)
# Train_lab1, Test_lab1 = train_test_split(ds_lab1, test_size=size, random_state=6)
# Train_lab2, Test_lab2 = train_test_split(ds_lab2, test_size=size, random_state=6)
# Train_lab3, Test_lab3 = train_test_split(ds_lab3, test_size=size, random_state=6)
# Train_lab4, Test_lab4 = train_test_split(ds_lab4, test_size=size, random_state=6)
# Train_lab5, Test_lab5 = train_test_split(ds_lab5, test_size=size, random_state=6)
# Train_lab6, Test_lab6 = train_test_split(ds_lab6, test_size=size, random_state=6)
# ds1 = np.concatenate((ds_lab0, ds_lab1), axis=0)
# ds2 = np.concatenate((ds_lab0, ds_lab1, ds_lab2, ds_lab3), axis=0)
# ds3 = ds_lab6
# ds4 = np.concatenate((ds_lab5, ds_lab6), axis=0)
# ds5 = np.concatenate((ds_lab0, ds_lab1, ds_lab2, ds_lab3, ds_lab4), axis=0)
# ds6 = ds
# print('Lenzi (2002) dataset', len(ds1))
# print('Marion (2006) dataset', len(ds2))
# print('Maso River dataset', len(ds3))
# print('Field dataset', len(ds4))
# print('Laboratory dataset', len(ds5))
# print('All dataset', len(ds6))
# '''Lenzi (2002) dataset'''
# X_ds1 = ds1[:, 0:4]
# Y_ds1 = ds1[:, 7]
# X_ds1 = X_ds1.astype('float')
# Y_ds1 = Y_ds1.astype('float')
# '''Marion (2006) dataset'''
# X_ds2 = ds2[:, 0:4]
# Y_ds2 = ds2[:, 7]
# X_ds2 = X_ds2.astype('float')
# Y_ds2 = Y_ds2.astype('float')
# '''Maso River dataset'''
# X_ds3 = ds3[:, 0:4]
# Y_ds3 = ds3[:, 7]
# X_ds3 = X_ds3.astype('float')
# Y_ds3 = Y_ds3.astype('float')
# '''Field dataset'''
# X_ds4 = ds4[:, 0:4]
# Y_ds4 = ds4[:, 7]
# X_ds4 = X_ds4.astype('float')
# Y_ds4 = Y_ds4.astype('float')
# '''Laboratory dataset'''
# X_ds5 = ds5[:, 0:4]
# Y_ds5 = ds5[:, 7]
# X_ds5 = X_ds5.astype('float')
# Y_ds5 = Y_ds5.astype('float')
# '''All dataset'''
# X_ds6 = ds6[:, 0:4]
# Y_ds6 = ds6[:, 7]
# X_ds6 = X_ds6.astype('float')
# Y_ds6 = Y_ds6.astype('float')
#
# with open('D:\\ZHR_ML\\' + save + '\\RF-model.csv', 'w') as f:
#     f.write('dataset' + ',' + 'R2' + ',' + 'RMSE' + ',' + 'MAE' + ',' + 'BIAS' + '\n')
# f.close()
# with open('D:\\ZHR_ML\\' + save + '\\GBDT-model.csv', 'w') as f:
#     f.write('dataset' + ',' + 'R2' + ',' + 'RMSE' + ',' + 'MAE' + ',' + 'BIAS' + '\n')
# f.close()
#
# '''Lenzi (2002) dataset'''
# pre_RF = model_RF.predict(X_ds1)  # 模型预测
# pre_RF = pre_RF.flatten()
# pre_GBDT = model_GBDT.predict(X_ds1)  # 模型预测
# pre_GBDT = pre_GBDT.flatten()
# y_obe = Y_ds1
# r1 = rs(pre_RF, y_obe)
# r2 = rs(pre_GBDT, y_obe)
# RMSE1 = rmse(pre_RF, y_obe)
# RMSE2 = rmse(pre_GBDT, y_obe)
# MAE1 = mae(pre_RF, y_obe)
# MAE2 = mae(pre_GBDT, y_obe)
# BIAS1 = bias(pre_RF, y_obe)
# BIAS2 = bias(pre_GBDT, y_obe)
# with open('D:\\ZHR_ML\\' + save + '\\RF-model.csv', 'a') as f:
#     f.write('Lenzi (2002) dataset' + ',')
#     f.write(str(r1) + ',' + str(RMSE1) + ',' + str(MAE1) + ',' + str(BIAS1) + '\n')
# f.close()
# with open('D:\\ZHR_ML\\' + save + '\\GBDT-model.csv', 'a') as f:
#     f.write('Lenzi (2002) dataset' + ',')
#     f.write(str(r2) + ',' + str(RMSE2) + ',' + str(MAE2) + ',' + str(BIAS2) + '\n')
# f.close()
# # 绘制预测值于实际值的散点图（展示训练效果）
# fig3, ax = plt.subplots(figsize=(5, 4), dpi=200)
# stan = np.linspace(0, 5, 10)
# s1 = plt.scatter(y_obe, pre_RF, marker='^', c='none', edgecolors='k')
# s2 = plt.scatter(y_obe, pre_GBDT, marker='s', c='k', edgecolors='k')
# plt.legend(handles=[s1, s2], labels=['S3-RF and' + '\n' + 'Lenzi (2002) dataset($R^2$'+'={:.3}'.format(r1) + ')',
#                                      'S3-GBDT and' + '\n' + 'Lenzi (2002) dataset($R^2$'+'={:.3}'.format(r2) + ')'],
#            loc='lower right', prop={'size': 8.5})  # frameon=False
# ax.plot(stan, stan, linestyle='-', color='r')
# x_major_locator = MultipleLocator(0.5)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlim((0, 4))
# plt.ylim((0, 4))
# plt.xlabel('Observed($y_s/H_s$)')
# plt.ylabel('Predicted($y_s/H_s$)')
# ax.text(.05, .92, '($z$)',
#         transform=ax.transAxes,
#         fontdict={'family': 'Times New Roman', 'size': 12},
#         zorder=4)
# plt.savefig('D:\\ZHR_ML\\' + save + '\\RG-Lenzi (2002) dataset.png',
#             dpi=300, bbox_inches="tight")  # 保存散点图
# plt.show()  # 展示散点图
#
# '''Marion (2006) dataset'''
# pre_RF = model_RF.predict(X_ds2)  # 模型预测
# pre_RF = pre_RF.flatten()
# pre_GBDT = model_GBDT.predict(X_ds2)  # 模型预测
# pre_GBDT = pre_GBDT.flatten()
# y_obe = Y_ds2
# r1 = rs(pre_RF, y_obe)
# r2 = rs(pre_GBDT, y_obe)
# RMSE1 = rmse(pre_RF, y_obe)
# RMSE2 = rmse(pre_GBDT, y_obe)
# MAE1 = mae(pre_RF, y_obe)
# MAE2 = mae(pre_GBDT, y_obe)
# BIAS1 = bias(pre_RF, y_obe)
# BIAS2 = bias(pre_GBDT, y_obe)
# with open('D:\\ZHR_ML\\' + save + '\\RF-model.csv', 'a') as f:
#     f.write('Marion (2006) dataset' + ',')
#     f.write(str(r1) + ',' + str(RMSE1) + ',' + str(MAE1) + ',' + str(BIAS1) + '\n')
# f.close()
# with open('D:\\ZHR_ML\\' + save + '\\GBDT-model.csv', 'a') as f:
#     f.write('Marion (2006) dataset' + ',')
#     f.write(str(r2) + ',' + str(RMSE2) + ',' + str(MAE2) + ',' + str(BIAS2) + '\n')
# f.close()
# # 绘制预测值于实际值的散点图（展示训练效果）
# fig4, ax = plt.subplots(figsize=(5, 4), dpi=200)
# stan = np.linspace(0, 5, 10)
# s1 = plt.scatter(y_obe, pre_RF, marker='^', c='none', edgecolors='k')
# s2 = plt.scatter(y_obe, pre_GBDT, marker='s', c='k', edgecolors='k')
# plt.legend(handles=[s1, s2], labels=['S3-RF and' + '\n' + 'Marion (2006) dataset($R^2$'+'={:.3}'.format(r1) + ')',
#                                      'S3-GBDT and' + '\n' + 'Marion (2006) dataset($R^2$'+'={:.3}'.format(r2) + ')'],
#            loc='lower right', prop={'size': 8.5})  # frameon=False
# ax.plot(stan, stan, linestyle='-', color='r')
# x_major_locator = MultipleLocator(0.5)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlim((0, 4))
# plt.ylim((0, 4))
# plt.xlabel('Observed($y_s/H_s$)')
# plt.ylabel('Predicted($y_s/H_s$)')
# ax.text(.05, .92, '($z$)',
#         transform=ax.transAxes,
#         fontdict={'family': 'Times New Roman', 'size': 12},
#         zorder=4)
# plt.savefig('D:\\ZHR_ML\\' + save + '\\RG-Marion (2006) dataset.png',
#             dpi=300, bbox_inches="tight")  # 保存散点图
# plt.show()  # 展示散点图
#
# '''Maso River dataset'''
# pre_RF = model_RF.predict(X_ds3)  # 模型预测
# pre_RF = pre_RF.flatten()
# pre_GBDT = model_GBDT.predict(X_ds3)  # 模型预测
# pre_GBDT = pre_GBDT.flatten()
# y_obe = Y_ds3
# r1 = rs(pre_RF, y_obe)
# r2 = rs(pre_GBDT, y_obe)
# RMSE1 = rmse(pre_RF, y_obe)
# RMSE2 = rmse(pre_GBDT, y_obe)
# MAE1 = mae(pre_RF, y_obe)
# MAE2 = mae(pre_GBDT, y_obe)
# BIAS1 = bias(pre_RF, y_obe)
# BIAS2 = bias(pre_GBDT, y_obe)
# with open('D:\\ZHR_ML\\' + save + '\\RF-model.csv', 'a') as f:
#     f.write('Maso River dataset' + ',')
#     f.write(str(r1) + ',' + str(RMSE1) + ',' + str(MAE1) + ',' + str(BIAS1) + '\n')
# f.close()
# with open('D:\\ZHR_ML\\' + save + '\\GBDT-model.csv', 'a') as f:
#     f.write('Maso River dataset' + ',')
#     f.write(str(r2) + ',' + str(RMSE2) + ',' + str(MAE2) + ',' + str(BIAS2) + '\n')
# f.close()
# # 绘制预测值于实际值的散点图（展示训练效果）
# fig5, ax = plt.subplots(figsize=(5, 4), dpi=200)
# stan = np.linspace(0, 5, 10)
# s1 = plt.scatter(y_obe, pre_RF, marker='^', c='none', edgecolors='k')
# s2 = plt.scatter(y_obe, pre_GBDT, marker='s', c='k', edgecolors='k')
# plt.legend(handles=[s1, s2], labels=['S3-RF and' + '\n' + 'Maso River dataset($R^2$'+'={:.3}'.format(r1) + ')',
#                                      'S3-GBDT and' + '\n' + 'Maso River dataset($R^2$'+'={:.3}'.format(r2) + ')'],
#            loc='lower right', prop={'size': 8.5})  # frameon=False
# ax.plot(stan, stan, linestyle='-', color='r')
# x_major_locator = MultipleLocator(0.5)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlim((0, 4))
# plt.ylim((0, 4))
# plt.xlabel('Observed($y_s/H_s$)')
# plt.ylabel('Predicted($y_s/H_s$)')
# ax.text(.05, .92, '($z$)',
#         transform=ax.transAxes,
#         fontdict={'family': 'Times New Roman', 'size': 12},
#         zorder=4)
# plt.savefig('D:\\ZHR_ML\\' + save + '\\RG-Maso River dataset.png',
#             dpi=300, bbox_inches="tight")  # 保存散点图
# plt.show()  # 展示散点图
#
# '''Field dataset'''
# pre_RF = model_RF.predict(X_ds4)  # 模型预测
# pre_RF = pre_RF.flatten()
# pre_GBDT = model_GBDT.predict(X_ds4)  # 模型预测
# pre_GBDT = pre_GBDT.flatten()
# y_obe = Y_ds4
# r1 = rs(pre_RF, y_obe)
# r2 = rs(pre_GBDT, y_obe)
# RMSE1 = rmse(pre_RF, y_obe)
# RMSE2 = rmse(pre_GBDT, y_obe)
# MAE1 = mae(pre_RF, y_obe)
# MAE2 = mae(pre_GBDT, y_obe)
# BIAS1 = bias(pre_RF, y_obe)
# BIAS2 = bias(pre_GBDT, y_obe)
# with open('D:\\ZHR_ML\\' + save + '\\RF-model.csv', 'a') as f:
#     f.write('Field dataset' + ',')
#     f.write(str(r1) + ',' + str(RMSE1) + ',' + str(MAE1) + ',' + str(BIAS1) + '\n')
# f.close()
# with open('D:\\ZHR_ML\\' + save + '\\GBDT-model.csv', 'a') as f:
#     f.write('Field dataset' + ',')
#     f.write(str(r2) + ',' + str(RMSE2) + ',' + str(MAE2) + ',' + str(BIAS2) + '\n')
# f.close()
# # 绘制预测值于实际值的散点图（展示训练效果）
# fig5, ax = plt.subplots(figsize=(5, 4), dpi=200)
# stan = np.linspace(0, 5, 10)
# s1 = plt.scatter(y_obe, pre_RF, marker='^', c='none', edgecolors='k')
# s2 = plt.scatter(y_obe, pre_GBDT, marker='s', c='k', edgecolors='k')
# plt.legend(handles=[s1, s2], labels=['S3-RF and' + '\n' + 'Field dataset($R^2$'+'={:.3}'.format(r1) + ')',
#                                      'S3-GBDT and' + '\n' + 'Field dataset($R^2$'+'={:.3}'.format(r2) + ')'],
#            loc='lower right', prop={'size': 8.5})  # frameon=False
# ax.plot(stan, stan, linestyle='-', color='r')
# x_major_locator = MultipleLocator(0.5)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlim((0, 4))
# plt.ylim((0, 4))
# plt.xlabel('Observed($y_s/H_s$)')
# plt.ylabel('Predicted($y_s/H_s$)')
# ax.text(.05, .92, '($a$)',
#         transform=ax.transAxes,
#         fontdict={'family': 'Times New Roman', 'size': 12},
#         zorder=4)
# plt.savefig('D:\\ZHR_ML\\' + save + '\\RG-Field dataset.png',
#             dpi=300, bbox_inches="tight")  # 保存散点图
# plt.show()  # 展示散点图
#
# '''Laboratory dataset'''
# pre_RF = model_RF.predict(X_ds5)  # 模型预测
# pre_RF = pre_RF.flatten()
# pre_GBDT = model_GBDT.predict(X_ds5)  # 模型预测
# pre_GBDT = pre_GBDT.flatten()
# y_obe = Y_ds5
# r1 = rs(pre_RF, y_obe)
# r2 = rs(pre_GBDT, y_obe)
# RMSE1 = rmse(pre_RF, y_obe)
# RMSE2 = rmse(pre_GBDT, y_obe)
# MAE1 = mae(pre_RF, y_obe)
# MAE2 = mae(pre_GBDT, y_obe)
# BIAS1 = bias(pre_RF, y_obe)
# BIAS2 = bias(pre_GBDT, y_obe)
# with open('D:\\ZHR_ML\\' + save + '\\RF-model.csv', 'a') as f:
#     f.write('Laboratory dataset' + ',')
#     f.write(str(r1) + ',' + str(RMSE1) + ',' + str(MAE1) + ',' + str(BIAS1) + '\n')
# f.close()
# with open('D:\\ZHR_ML\\' + save + '\\GBDT-model.csv', 'a') as f:
#     f.write('Laboratory dataset' + ',')
#     f.write(str(r2) + ',' + str(RMSE2) + ',' + str(MAE2) + ',' + str(BIAS2) + '\n')
# f.close()
# # 绘制预测值于实际值的散点图（展示训练效果）
# fig5, ax = plt.subplots(figsize=(5, 4), dpi=200)
# stan = np.linspace(0, 5, 10)
# s1 = plt.scatter(y_obe, pre_RF, marker='^', c='none', edgecolors='k')
# s2 = plt.scatter(y_obe, pre_GBDT, marker='s', c='k', edgecolors='k')
# plt.legend(handles=[s1, s2], labels=['S3-RF and' + '\n' + 'Laboratory dataset($R^2$'+'={:.3}'.format(r1) + ')',
#                                      'S3-GBDT and' + '\n' + 'Laboratory dataset($R^2$'+'={:.3}'.format(r2) + ')'],
#            loc='lower right', prop={'size': 8.5})  # frameon=False
# ax.plot(stan, stan, linestyle='-', color='r')
# x_major_locator = MultipleLocator(0.5)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlim((0, 4))
# plt.ylim((0, 4))
# plt.xlabel('Observed($y_s/H_s$)')
# plt.ylabel('Predicted($y_s/H_s$)')
# ax.text(.05, .92, '($b$)',
#         transform=ax.transAxes,
#         fontdict={'family': 'Times New Roman', 'size': 12},
#         zorder=4)
# plt.savefig('D:\\ZHR_ML\\' + save + '\\RG-Laboratory dataset.png',
#             dpi=300, bbox_inches="tight")  # 保存散点图
# plt.show()  # 展示散点图
#
# '''All dataset'''
# pre_RF = model_RF.predict(X_ds6)  # 模型预测
# pre_RF = pre_RF.flatten()
# pre_GBDT = model_GBDT.predict(X_ds6)  # 模型预测
# pre_GBDT = pre_GBDT.flatten()
# y_obe = Y_ds6
# r1 = rs(pre_RF, y_obe)
# r2 = rs(pre_GBDT, y_obe)
# RMSE1 = rmse(pre_RF, y_obe)
# RMSE2 = rmse(pre_GBDT, y_obe)
# MAE1 = mae(pre_RF, y_obe)
# MAE2 = mae(pre_GBDT, y_obe)
# BIAS1 = bias(pre_RF, y_obe)
# BIAS2 = bias(pre_GBDT, y_obe)
# with open('D:\\ZHR_ML\\' + save + '\\RF-model.csv', 'a') as f:
#     f.write('All dataset' + ',')
#     f.write(str(r1) + ',' + str(RMSE1) + ',' + str(MAE1) + ',' + str(BIAS1) + '\n')
# f.close()
# with open('D:\\ZHR_ML\\' + save + '\\GBDT-model.csv', 'a') as f:
#     f.write('All dataset' + ',')
#     f.write(str(r2) + ',' + str(RMSE2) + ',' + str(MAE2) + ',' + str(BIAS2) + '\n')
# f.close()
# # 绘制预测值于实际值的散点图（展示训练效果）
# fig5, ax = plt.subplots(figsize=(5, 4), dpi=200)
# stan = np.linspace(0, 5, 10)
# s1 = plt.scatter(y_obe, pre_RF, marker='^', c='none', edgecolors='k')
# s2 = plt.scatter(y_obe, pre_GBDT, marker='s', c='k', edgecolors='k')
# plt.legend(handles=[s1, s2], labels=['S3-RF and' + '\n' + 'All dataset($R^2$'+'={:.3}'.format(r1) + ')',
#                                      'S3-GBDT and' + '\n' + 'All dataset($R^2$'+'={:.3}'.format(r2) + ')'],
#            loc='lower right', prop={'size': 8.5})  # frameon=False
# ax.plot(stan, stan, linestyle='-', color='r')
# x_major_locator = MultipleLocator(0.5)
# y_major_locator = MultipleLocator(0.5)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlim((0, 4))
# plt.ylim((0, 4))
# plt.xlabel('Observed($y_s/H_s$)')
# plt.ylabel('Predicted($y_s/H_s$)')
# ax.text(.05, .92, '($z$)',
#         transform=ax.transAxes,
#         fontdict={'family': 'Times New Roman', 'size': 12},
#         zorder=4)
# plt.savefig('D:\\ZHR_ML\\' + save + '\\RG-All dataset.png',
#             dpi=300, bbox_inches="tight")  # 保存散点图
# plt.show()  # 展示散点图