import numpy as np
import matplotlib.pylab as plt
import jupyter
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

#读取txt并切割数据集
data=load_svmlight_file('housing_scale')
x_train, x_test, y_train, y_test=train_test_split(data[0],data[1], test_size=0.33)

#初始化参数
learning_rate=0.00001
iter_number=10000
init_w=np.zeros(shape=[14,1])
x_train = x_train.todense()
x_test = x_test.todense()
Y_train=np.mat(y_train)
Y_test=np.mat(y_test)


matrix1=np.ones(shape=[339,1])
matrix2=np.ones(shape=[167,1])
X_train=np.hstack((x_train,matrix1))
X_test=np.hstack((x_test,matrix2))

#梯度求解
gradient=-np.dot(X_train.T,Y_train.T)+np.dot(np.dot(X_train.T, X_train),init_w)
w=init_w

#循环？
train_list = []
test_list = []
for loss in range(iter_number):
    w=w-gradient*learning_rate
    gradient=-np.dot(X_train.T,Y_train.T)+np.dot(np.dot(X_train.T, X_train),w)
    #求损失函数
    loss_train=np.dot((np.dot(X_train,w)-Y_train.T).T,(np.dot(X_train,w)-Y_train.T))
    loss_test=np.dot((np.dot(X_test,w)-Y_test.T).T,(np.dot(X_test,w)-Y_test.T))
    l1=loss_train.tolist()
    l2=loss_test.tolist()
    train_list.append(l1[0][0])
    test_list.append(l2[0][0])

plt.plot(np.arange(0, iter_number), train_list)
plt.plot(np.arange(0, iter_number), test_list)
plt.show()


