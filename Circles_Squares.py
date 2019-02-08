from sklearn import tree
import math


# classification of Circles vs Squares
# syntax for features
# Features =Has a radius, Has an Area
features = [[1, (1*math.pi)], [1, (2*math.pi)], [1, (3*math.pi)], [0, (1*1)], [0, (2*2)], [0, (3*3)]]
# Labeled Training Data
# labels = [["Has Radius=1".Area of a circle],["Has Radius=0,Length * Height Area of a square]
# 0=Circle, 1=Square
labels = [0, 0, 0, 1, 1, 1]

# create your classifier for the training data
# sci-kit learn Documentation https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
clf = tree.DecisionTreeClassifier()

# Fit is finding patterns in the data
clf = clf.fit(features, labels)

# input unknown data
# HP=160, Number of Seats =7
# Prediction should be a minivan [0]
print(clf.predict([[1, (4*math.pi)]]))


# HP=600, number of seats =2
# Prediction should be a sports car [0]
print(clf.predict([[0, (4*4)]]))

