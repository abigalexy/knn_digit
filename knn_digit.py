import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 读取数据
data = pd.read_csv('手写数字识别.csv')
x = data.iloc[:,1:]
y = data.iloc[:,0]
print(Counter(y))

# 图片显示
digit = x.iloc[1000].values
img = digit.reshape(28,28)
plt.imshow(img,cmap='gray')
plt.imsave('digit.png',img)
plt.show()

# 数据归一化处理
x = x/255
# 数据集划分
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=22)

# 实例化模型
# model = KNeighborsClassifier(n_neighbors=11)
# 模型训练
# model.fit(x_train,y_train)
# 模型预测
# img = plt.imread('digit.png')
# img = img[:,:,1].reshape(1,-1)/255
# y_predict = model.predict(x_test)
# print(y_predict)
# 模型评估
# print(model.score(x_test,y_test))
# print(accuracy_score(y_predict,y_test))
# 模型保存
# joblib.dump(model,'knn.pth')

# 模型加载
knn = joblib.load('knn.pth')
print(knn.score(x_test,y_test))
img = plt.imread('digit.png')
img = img[:,:,1].reshape(1,-1)
print(knn.predict(img))
