import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import ensemble

csvfile = "D:\\Deep Learning\\Titanic\\Data\\train.csv"
csvfile2 = "D:\\Deep Learning\\Titanic\\Data\\test.csv"

df = pd.read_csv(csvfile, header=0)
df_test = pd.read_csv(csvfile2, header=0)
#df.info()

cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)
df_test = df_test.drop(cols, axis=1)
ori_df = df

df = df.dropna()
#df.info()
dummies = []
dummies2 = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
    dummies.append(pd.get_dummies(ori_df[col]))
    dummies2.append(pd.get_dummies(df_test[col]))
titanic_dummies = pd.concat(dummies, axis=1)
titanic_dummies2 = pd.concat(dummies2, axis=1)
titanic_dummies.info()

df = pd.concat((ori_df, titanic_dummies), axis=1)
df2 = pd.concat((df_test, titanic_dummies2), axis=1)
df = df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
df2= df2.drop(['Pclass', 'Sex', 'Embarked'], axis=1)

df['Age'] = df['Age'].interpolate()
df2['Age'] = df2['Age'].interpolate()
df2['Fare'] = df2['Fare'].interpolate()

df2.info()

X = df.values
y = df['Survived'].values
print(X)
X = np.delete(X, 1, axis=1)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
'''
##############################DecisionTreeClassifier=0.7910447761194029##########################
clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
#################################################################################################
'''


clf = ensemble.GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
X_results = df2.values
y_results = clf.predict(X_results)

output = np.column_stack((X_results[:, 0], y_results))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID','Survived'])
df_results.to_csv('D:\\Deep Learning\\Titanic\\titanic_results.csv', index=False)
