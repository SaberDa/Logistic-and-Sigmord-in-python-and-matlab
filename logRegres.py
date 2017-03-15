# -*- coding: utf-8 -*-

from numpy import *

# 打开文件读取数据
# 每行前两个值分别为X1、X2，第三个值是数据对应的类别标签
# 第0维特征X0 = 1
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))


# 梯度上升算法
# dataMatIn: 2维Numpy数组，每列分别代表每个不同的特征，每行代表每个训练样本
# classLabels: 类别标签，一组一维行向量
def gradAscent(dataMatIn, classLabels):
    # 转换为矩阵类型
    dataMatrix = mat(dataMatIn)
    # 为便于计算，将类别标签转置
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001   # 目标移动的步长
    maxCycles = 500     # 迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        # 矩阵相乘
        h = sigmoid(dataMatrix * weights)
        # 计算真实类别与预测类型的差值
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 画出数据集合Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c = 'red', marker='s')
    ax.scatter(xcord2, ycord2, s=10, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    # 最佳拟合直线
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法
# 代码与前面的梯度上升很像，但是这里计算weights时是用的向量，而非数值；而且前者没有使用矩阵转换，使用的是Numpy数组
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 随机梯度上升改进版
# 规定迭代次数为150次
def stocGradAscent1(dataMatrix, classLabels, numInter = 150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numInter):
        dataIndex = range(m)
        for i in range(m):
            # alpha 在每次迭代时需要调整，一避免参数的严格下降
            alpha = 4/(1.0 + j + i) + 0.01
            # 随机选取更新，以减少周期性的波动
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 删除此次选取的随机值，方便下次迭代
            del(dataIndex[randIndex])
    return weights






