import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
import time

trainData = pd.read_csv("./input/train.csv").values
testData = pd.read_csv("./input/test.csv").values

# clf = naive_bayes.BernoulliNB()
clf = naive_bayes.GaussianNB()

#Training dataset
xTrain = trainData[0:, 1:]
yTrain = trainData[0:,0]
t1 = time.time()
clf.fit( X=xTrain, y=yTrain)

# depth of the decision tree
#print('Depth of the Decision Tree :', clf.get_depth())

# Predict value
predtictY = clf.predict(testData)
t2 = time.time()
print("training time:", str(t2-t1))
# Create submission file
df_sub = pd.DataFrame(list(range(1,len(testData)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = predtictY
# df_sub.to_csv("./output/sklearn_BernoulliNaiveBayes.csv",index=False)
df_sub.to_csv("./output/sklearn_GaussianNaiveBayes.csv",index=False)
