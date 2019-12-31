# For Python 2 / 3 compatability
from __future__ import print_function

from csv import reader
from math import sqrt
import pandas as pd
from random import randrange
import time


def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# get training data
filename = 'train.csv'
test_filename = 'test.csv'
filename_dir = 'input/'
dataset = load_csv(filename_dir + filename)
testdata = load_csv(filename_dir + test_filename)

# ignore first column as it holds the actual class of digit
training_data = dataset[1:]




# Column labels.
# These are used only to print the tree.
header = dataset[0]

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def class_counts_mnist(rows):
    counts = {}
    for row in rows:
        # First entry in row is label
        label = row[0]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    #print("==> Counts:", counts)
    return counts


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            # the row's value of the column was greater than or equal to the questions value
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows



def gini(rows):
    counts = class_counts_mnist(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns
    #print("n_features:", n_features)

    for col in range(1,n_features):  # for each feature
        # for each iteration this is the set of all values of a specific column, eg, All pixels number 0
        values = set([row[col] for row in rows])  # unique values in the column
        for val in values:  # for each value

            # Create a question object for each val under a column, holding the val and the col number
            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts_mnist(rows)


class Decision_Node:

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_single_random_tree(rows, random_features, max_depth, min_size, depth):

    gain, question = find_best_split(rows)

    # Check if gain = 0 ==> means we are at a leafnode
    if gain == 0:
        return Leaf(rows)

    # Check if depth is equal to max depth:
    if depth >= max_depth:
        # return a leaf
        return Leaf(rows)


    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Check sizes of true and false rows, if smaller than min size ==> not splitting them further
    if (len(true_rows) > min_size):
        # Recursively build the true branch.
        true_branch = build_single_random_tree(true_rows, random_features, max_depth, min_size, depth+1)
    else:
        true_branch = Leaf(true_rows)
    
    if (len(false_rows) > min_size):
        # Recursively build the false branch.
        false_branch = build_single_random_tree(false_rows, random_features, max_depth, min_size, depth+1)
    else:
        false_branch = Leaf(false_rows)
    

    return Decision_Node(question, true_branch, false_branch)

def select_random_rows(rows, random_dataset_size):
    random_data_set = []
    for x in range(random_dataset_size):
        #from 1, as label is in column 0
        random_row_index = randrange(1, len(rows))
        random_data_set.append(rows[random_row_index])
    return random_data_set

def build_random_trees(rows, n_features, max_depth, min_size, n_trees, random_dataset_size):
    """
    For n_trees
    Select random n_features from rows
    make them max_depth deep and with a minimum of min_size
    """
    trees = []
    for tree_number in range(n_trees):
        print("Building tree number:", tree_number, "of", n_trees)
        # Select random dataset from original dataset
        random_dataset = select_random_rows(rows, random_dataset_size)

        # Select random features (columns)
        random_features = []
        for random_feature in range (n_features):
            # generate random index number to pick column
            random_column = randrange(len(rows))
            random_features.append(random_column)
        # generate the random tree with randomly picked features (columns) and a random dataset
        random_tree = build_single_random_tree(random_dataset, random_features, max_depth, min_size, 1)
        # add to list of trees
        trees.append(random_tree)
    return trees

def build_tree(rows, treecounter):

    print("Treecounter:", treecounter)
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, treecounter+1)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, treecounter+1)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def leaf_pred(counts):
    #print(counts)
    if len(counts.keys()) == 1:
        for key in counts.keys():
            #print("return: ", key)
            return key
    else:
        return(dict_max_key(counts))

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def dict_max_key(counts):
    #print("Finding max key in:", counts)
    multiple_keys = []
    b_val = 0
    for key in counts.keys():
        val = counts[key]
        if val > b_val:
            b_val = val
            multiple_keys = [key]
        elif val == b_val:
            multiple_keys.append(key)
    
    # multiple_keys now holds all keys with the dict's max value
    # pick a random winner
    if len(multiple_keys) == 1:
        #only one key
        return multiple_keys[0]
    else:
        # return a random index of multiple keys
        return multiple_keys[randrange(0, len(multiple_keys))]



if __name__ == '__main__':
    t1 = time.time()
    forest = build_random_trees(training_data, 20, 5, 5, 20, round(len(training_data)*0.10))

    for tree in forest:
        print_tree(tree)

    predictions = []
    for row in testdata[1:]:
        if(len(predictions) % 100 == 0):
            print("predicting:", len(predictions))
        # collect each tree's prediction
        forest_predictions = {}
        for tree in forest:
            tree_prediction = leaf_pred(classify(row[1:], tree))
            if tree_prediction in forest_predictions.keys():
                forest_predictions[tree_prediction] = forest_predictions[tree_prediction] + 1
            else:
                forest_predictions[tree_prediction] = 1
        
        # find most common in forest
        predictions.append(dict_max_key(forest_predictions))

    t2 = time.time()
    print("training time:", str(t2-t1))

    #print(predictions)
    """for row in testdata:
        print ("Actual: %s. Predicted: %s" %
               (row[0], print_leaf(classify(row, my_tree))))"""
    
    #Local check accuracy
    """
    correct = 0
    for x in range(len(local_test)):
        if predictions[x] == local_test[x][0]:
            correct = correct +1
    print("Accuracy:", correct/len(local_test)*100, "%")"""

    
    # Create submission file
    print("Len(testdata):", len(testdata))
    print("Len(predictions):", len(predictions))
    print("prediction[0]:", predictions[0])
    print("testdata[0]:", testdata[0])
    
    df_sub = pd.DataFrame(list(range(1,len(testdata))))
    df_sub.columns = ["ImageID"]
    df_sub["Label"] = predictions
    df_sub.to_csv("output/rand_tree.csv",index=False)
    
