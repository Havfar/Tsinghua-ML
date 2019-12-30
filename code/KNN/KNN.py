from collections import Counter

import pandas as pd
import numpy as np



class KNN():
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=7):
        dists = self.compute_distances(X)

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            k_closest_y = []
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            # find k nearest lables
            k_closest_y = labels[:k]
            c = Counter(k_closest_y)
            y_pred[i] = c.most_common(1)[0][0]

        return(y_pred)

    def compute_distances(self, X):
        dot_pro = np.dot(X, self.X_train.T)
        sum_square_test = np.square(X).sum(axis = 1)
        sum_square_train = np.square(self.X_train).sum(axis = 1)
        distance = np.sqrt(-2 * dot_pro + sum_square_train + np.matrix(sum_square_test).T)

        return(distance)

trainData = pd.read_csv("../../input/train.csv").values
testData = pd.read_csv("../../input/test.csv").values
print(trainData.shape)
#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]
model = KNN()
model.fit(xTrain, yTrain)
# Predict value
predtictY = model.predict(testData)
# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY
df_sub.to_csv("../../output/9-NN.csv",index=False)
