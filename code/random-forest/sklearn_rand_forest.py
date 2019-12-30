import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier



trainData = pd.read_csv("./input/train.csv").values
testData = pd.read_csv("./input/test.csv").values
print(trainData.shape)
#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]
model = RandomForestClassifier(n_estimators=100)
t1 = time.time()
model.fit(xTrain, yTrain)
# Predict value
predtictY = model.predict(testData)
t2 = time.time()
print("training time:", str(t2-t1))
# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY
df_sub.to_csv("./output/randomForestn100sk.csv",index=False)
