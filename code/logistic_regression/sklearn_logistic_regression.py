import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression



trainData = pd.read_csv("./input/train.csv").values
testData = pd.read_csv("./input/test.csv").values
print(trainData.shape)
#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]
#model = LogisticRegression()
model = LogisticRegression(
    penalty='l1', solver='saga', tol=0.1
)
t1 = time.time()
model.fit(xTrain, yTrain)

# Predict value
predtictY = model.predict(testData)
t2 = time.time()
print("Time training:", str(t2-t1))
# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY
df_sub.to_csv("./output/new_sklearn_LR.csv",index=False)
