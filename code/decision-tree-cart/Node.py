class Node:

    def __init__(self, gini, num_samples, num_samples_per_class, predict_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predict_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        print(predict_class)
    # def __init__(self):
    #     print("YAY")


    def debug(self):
        pass

    def _debug_aux(self):
        pass

