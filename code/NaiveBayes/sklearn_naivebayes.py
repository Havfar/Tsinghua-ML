import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes

trainData = pd.read_csv("../../input/train.csv").values
testData = pd.read_csv("../../input/test.csv").values

clf = naive_bayes.BernoulliNB()

#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]

clf.fit( X=xTrain, y=yTrain)

# depth of the decision tree
#print('Depth of the Decision Tree :', clf.get_depth())

# Predict value
predtictY = clf.predict(testData)

# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY
df_sub.to_csv("../../output/sklearn_BernoulliNaiveBayes.csv",index=False)
