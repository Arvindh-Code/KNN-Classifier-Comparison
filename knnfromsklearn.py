from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as num
import pandas as panda

#using sklearn package, the accuracy of knn with kfold croess validation has been done for the car dataset
kvalues=[3,5,7,9,11,13,15,17,19,21]
listvalue = panda.read_csv('/Users/aravindh/Downloads/car+evaluation/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
listvalue[['buying','maint','doors','persons','lug_boot','safety']] = listvalue[['buying','maint','doors','persons','lug_boot','safety']].apply(LabelEncoder().fit_transform)
xsetvalue=listvalue.drop(columns=['class'])
ysetvalue=listvalue['class']
x_set_train,x_set_test,y_set_train,y_set_test=train_test_split(xsetvalue,ysetvalue,test_size=0.3,random_state=42)
standard_scale = StandardScaler()
x_set_train[['doors','persons']]=standard_scale.fit_transform(x_set_train[['doors','persons']])
x_set_test[['doors','persons']]= standard_scale.transform(x_set_test[['doors','persons']])
sklearn_knn_accuracy = []
for k in kvalues:
    custom_knn = KNeighborsClassifier(n_neighbors=k)
    custom_knn.fit(x_set_train, y_set_train)
    accuracy_custom = accuracy_score(y_set_test, custom_knn.predict(x_set_test))
knn = KNeighborsClassifier()
for k in kvalues:
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(knn, xsetvalue, ysetvalue, cv=kfold, scoring='accuracy')
    sklearn_knn_accuracy.append(num.mean(accuracy_scores))
mean=0
print("Accuracy Scores car dataset:")
with open('car_sklearn.txt', 'w') as f:
    for k, accuracy in zip(kvalues, sklearn_knn_accuracy):
        mean += accuracy * 100
        f.write(f"{accuracy}\n")
print("Car qaccuracy ",mean/10)
print("\n")




#using sklearn package, the accuracy of knn with kfold croess validation has been done for the hayes roth dataset
list_value = panda.read_csv('/Users/aravindh/Downloads/hayes+roth/hayes-roth.data', names=['name', 'hobby', 'age', 'educational_level', 'marital_status', 'class'])
list_value[['hobby','age','educational_level','marital_status','class']]=list_value[['hobby','age','educational_level','marital_status','class']].apply(LabelEncoder().fit_transform)
x_set=list_value.drop(columns=['class'])
y_set=list_value['class']
strandard_scale=StandardScaler()
x_set[['name','age','educational_level']] = strandard_scale.fit_transform(x_set[['name','age','educational_level']])
sklearn_knn_accuracy = []
knn = KNeighborsClassifier()
for k in kvalues:
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    score = cross_val_score(knn, x_set, y_set, cv=kfold, scoring='accuracy')
    sklearn_knn_accuracy.append(num.mean(score))

mean=0
print("Accuracy Scores Hayens-roth dataset:")
with open('hayens_roth_sklearn.txt', 'w') as f:
    for k, accuracy in zip(kvalues, sklearn_knn_accuracy):
        mean += accuracy * 100
        f.write(f"{accuracy}\n")
print("Hayens-roth qaccuracy ",mean/10)
print("\n")


#using sklearn package, the accuracy of knn with kfold croess validation has been done to the breast cancer dataset
age={'10-19':15,'20-29':25,'30-39':35,'40-49':45,'50-59':55,'60-69':65,'70-79':75,'80-89':85,'90-99':95}
tumor={'0-4':2,'5-9':7,'10-14':12,'15-19':17,'20-24':22,'25-29':27,'30-34':32,'35-39':37,'40-44':42,'45-49':47,'50-54':52,'55-59':57}
nodes={'0-2':1,'3-5':4,'6-8':7,'9-11':10,'12-14':13,'15-17':16,'18-20':19,'21-23':22,'24-26':25,'27-29':28,'30-32':31,'33-35':34,'36-39':37}
setvalue=panda.read_csv('/Users/aravindh/Downloads/breast+cancer/breast-cancer.data',
                        names=['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad', 'irradiat'])
setvalue[['menopause','node-caps','breast','breast-quad','irradiat']]=setvalue[['menopause','node-caps','breast','breast-quad', 'irradiat']].apply(LabelEncoder().fit_transform)
setvalue['age']=setvalue['age'].map(age)
setvalue['tumor-size']=setvalue['tumor-size'].map(tumor)
setvalue['inv-nodes']=setvalue['inv-nodes'].map(nodes)
setvalue[['age','tumor-size','inv-nodes','deg-malig']]=StandardScaler().fit_transform(setvalue[['age','tumor-size','inv-nodes','deg-malig']])
x_set = setvalue.drop('Class', axis=1)
y_set = setvalue['Class']
set_x_train,set_x_test,set_y_train,set_y_test=train_test_split(x_set,y_set,test_size=0.3,random_state=4)
sklearn_knn_accuracy = []
for k in kvalues:
    custom_knn = KNeighborsClassifier(n_neighbors=k)
    custom_knn.fit(set_x_train, set_y_train)
    acc = accuracy_score(set_y_test, custom_knn.predict(set_x_test))
knn = KNeighborsClassifier()
for k in kvalues:
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    score = cross_val_score(knn, x_set, y_set, cv=kfold, scoring='accuracy')
    sklearn_knn_accuracy.append(num.mean(score))
mean=0
print("Accuracy Scores breast_cancer dataset:")
with open('breast_cancer_sklearn.txt', 'w') as f:
    for k, accuracy in zip(kvalues, sklearn_knn_accuracy):
        mean += accuracy * 100
        f.write(f"{accuracy}\n")
print("Breast Cancer accuracy ",mean/10)
