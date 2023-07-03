import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb
from sklearn import model_selection
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
#Data Visualization
url='https://raw.githubusercontent.com/nithish1018/Data-set-for-machine-learning/main/weight.csv'
df = pd.read_csv(url)
print(df)
pt.xlabel("height")
pt.ylabel("weight")
df1 = df[df.target==0]
df2 = df[df.target==1]
df3 = df[df.target==2]
#Scatter Diagram
pt.scatter(df1["height"], df1["weight"], color = "red", marker = "+")
pt.scatter(df2["height"], df2["weight"], color = "green", marker = "*")
pt.scatter(df3["height"], df3["weight"], color = "black", marker = ".")
pt.show()
#Experiance
x = df.drop(["target"], axis = "columns")
y = df["target"]
xtrain,xtest,ytrain,ytest = sklearn_train_test_split(x,ytest_size = 0.2, random_state = 1)
print(xtrain)
print(ytrain)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(xtrain,ytrain)
#Task
ypredict = knn.predict(xtest)
cm = confusion_matrix(ytest,ypredict)
print("confusion matrix = ",cm)
pt.figure(figsize = (10,5))
sb.heatmap(cm, annot = true)
pt.xlabel("Predicted Value")
pt.ylabel("Actual value from Dataset")
pt.show()
#End User Input
print("Enter Height and Weight")
h = int(input())
w = int(input())
data = {"height" : [h], "weight" : [w]}
k = pd.DataFrame(data)
pt = knn.predict(k[["height","weight"]])
print("predicted target = ", pt)
#Performance
acc = knn.score(xtest,ytest)
acc = int(round(acc,2)*100)
print("accuracy = ", acc, "%")
