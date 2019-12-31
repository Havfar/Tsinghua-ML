import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

trainData = pd.read_csv("../../input/train.csv").values
testData = pd.read_csv("../../input/test.csv").values
print(trainData.shape)
#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]
model = KNeighborsClassifier()
model.fit(xTrain, yTrain)
# Predict value
predtictY = model.predict(testData)
# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY
df_sub.to_csv("../../output/sklearn_KNN.csv",index=False)
