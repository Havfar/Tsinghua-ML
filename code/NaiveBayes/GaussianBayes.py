import numpy as np
import pandas as pd
from scipy.stats import norm


class GuassianNB():

    def fit(self, X_train, y_train):
        self.min_std = 0.000001
        self.nclass = np.unique(y_train).shape[0]
        self.class_labels = np.unique(y_train)
        self.nfeature = X_train.shape[1]
        self.means = np.zeros((self.nclass, self.nfeature))
        self.stds = np.zeros((self.nclass, self.nfeature))
        self.log_py = np.zeros(self.nclass)
        for i in range(self.nclass):
            # Get the boolean vector to filter for y = i
            mask = [l == self.class_labels[i] for l in y_train]
            self.means[i] = np.nanmean(X_train[mask], axis=0)
            # To avoid devide by 0/very small value issue, add a min for standard deviation to min_std = 0.00000001
            self.stds[i] = np.clip(np.nanstd(X_train[mask], axis=0), self.min_std, None)
            # calculate p(y=i)
            self.log_py[i] = np.log(np.sum(mask) / len(y_train))

    def predict(self, X_test):
        samples = X_test.shape[0]
        log_py_on_x = np.zeros((samples, self.nclass))
        for i in range(self.nclass):
            log_py_on_x[:, i] = self.log_py[i] + np.nansum(np.log(norm.pdf(X_test, self.means[i], self.stds[i])),
                                                           axis=1)
        label = self.class_labels[np.argmax(log_py_on_x, axis=1)]
        return label


if __name__ == "__main__":
    import pandas as pd
    print("LOADING DATA")
    #load data
    train_data = pd.read_csv("../../input/train.csv").values
    test_data = pd.read_csv("../../input/test.csv").values

    clf = GuassianNB()

    # Training dataset
    train_data = train_data
    xTrain = train_data[0:, 1:]
    yTrain = train_data[0:, 0]

    clf.fit(xTrain, yTrain)

    # Predict value
    predtictY = clf.predict(test_data)

    # Create submission file
    df_sub = pd.DataFrame(list(range(1, len(test_data) + 1)))
    df_sub.columns = ["ImageID"]
    df_sub["Label"] = predtictY
    df_sub.to_csv("../../output/havlearn_GaussianNaiveBayes.csv", index=False)