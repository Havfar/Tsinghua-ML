import pandas as pd
import numpy as np
import time
from sklearn import datasets

class LogisticRegression:
    def __init__(self, lr = 0.01, num_iter = 100, fit_intercept = True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0],1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def __loss(self, h, y):
        return (-y*np.log(h) - (1-y)*np.log(1-h)).mean()

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.1):
        test = self.predict_prob(X)
        print(test[0:10])
        return test

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        self.theta  = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            if(i % 10 == 0):
                print("i: ",i,"/", self.num_iter)
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h-y))/y.size
            self.theta -= self.lr * gradient


trainData = pd.read_csv("./input/train.csv").values
testData = pd.read_csv("./input/test.csv").values
trainData1 = datasets.load_digits()
print(trainData1.images.shape)
print(trainData.shape)
#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]
t1 = time.time()
model = LogisticRegression()
model.fit(xTrain, yTrain)

# Predict value
predtictY = model.predict(testData)
t2 = time.time()
print("training time:", str(t2-t1))
# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY
df_sub.to_csv("./output/havlearn_LR.csv",index=False)
