from __future__ import print_function
# Creating a toy dataset, has colour, fruit diameter and fruit name
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
    ['Orange', 2, 'Mandarin'],
    ['Orange', 3, 'Mandarin']
]
header = ["Colour", "Diameter", "Label"]


def unique_vals(rows, col):  # To find the unique values for a column in a dataset
    return set([row[col] for row in rows])


def class_counts(rows):  # Counts each type of example in a dataset
    counts = {}
    for row in rows:
        label = row[-1]  # Label is the last column
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def isnumeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question():  # For partioning a dataset
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):  # Compare feature value in example to the question
        val = example[self.column]
        if isnumeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):  # To print the question in readable format
        condition = "=="
        if isnumeric(self.value):
            condition = ">="
        print(header[self.column] + condition + str(self.value))


""" Question(1,3) will give us "Is diameter >= 3?"
Question(0, "green") gives us "Is colour == green?" """


def partition(rows, question):  # For partitionaing the dataset
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


"""If query matches the question then adds it to true rows and all others to the false rows"""


def gini(rows):  # For calculating the gini impurity for a row
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


"""This returns 0 when no mixing
0.5 if equal mixing
Any no. b/w 0-1 if mixing"""


def info_gain(left, right, curr_unc):  # For info gain
    p = float(len(left)) / (len(left) + len(right))
    return curr_unc - p * gini(left) - (1 - p) * gini(right)


""" Info gain by partitioning on green?
true_rows, false_rows = partition(training_data, Question(0, 'Green'))
info_gain(true_rows, false_rows, current_uncertainty)
This gives us 0.139"""

"""Info gain by partitioning on red?
true_rows, false_rows = partition(training_data, Question(0,'Red'))
info_gain(true_rows, false_rows, current_uncertainty)
This gives us 0.373, greater than that for green."""


def best_split(rows):  # To find best question to ask
    best_gain = 0
    best_question = None
    curr_unc = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            # Split the data
            true_rows, false_rows = partition(rows, question)
            # Skip the splitting if no division is needed
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            # Calculating the info gain from this split
            gain = info_gain(true_rows, false_rows, curr_unc)
            if (gain >= best_gain):
                best_gain, best_question = gain, question
    return best_gain, best_question


class Leaf:  # Leaf node to classify data
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):

        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


""" The following is a recursive function. What is it exactly?
Ans:  Recursion is a way of programming or coding a problem, in which a function calls itself one or more times in its body. Usually, it is returning the return value of this function call. If a function definition fulfils the condition of recursion, we call this function a recursive function."""


def build_tree(rows):  # Actual tree builder
    gain, question = best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=" "):  # Preset tree printing function
    # Base case
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Question at this node
    # return spacing + str(node.question)

    # Calling function recursively for true branch
    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + " ")
    # Calling function recursively for true branch
    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + " ")


my_tree = build_tree(training_data)
print_tree(my_tree)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


testing_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

for row in testing_data:
    print ("Actual: %s. Predicted: %s" %
           (row[-1], print_leaf(classify(row, my_tree))))
