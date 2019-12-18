import pandas as pd
import numpy as np
from sklearn.linear_model import logistic



trainData = pd.read_csv("../../input/train.csv").values
testData = pd.read_csv("../../input/test.csv").values
print(trainData.shape)
#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]
model = logistic.LogisticRegression()
model.fit(xTrain, yTrain)
# Predict value
predtictY = model.predict(testData)
# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY
df_sub.to_csv("../../output/havlearn_LR.csv",index=False)
