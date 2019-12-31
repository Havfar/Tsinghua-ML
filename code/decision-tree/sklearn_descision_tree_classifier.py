import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import time

test = pd.read_csv("./input/train.csv")
trainData = pd.read_csv("./input/train.csv").values
testData = pd.read_csv("./input/test.csv").values
clf = DecisionTreeClassifier()

#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]


# depth of the decision tree

t1 = time.time()
clf.fit(xTrain, yTrain)


# Predict value
predtictY = clf.predict(testData)
t2 = time.time()

# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY

df_sub.to_csv("./output/sklearn_decission_tree_classifier.csv",index=False)
print("training time:", str(t2-t1))