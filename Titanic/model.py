import pandas as pd
import numpy as np
#from sklearn import tree
#from sklearn.model_selection import train_test_split
from sklearn import ensemble

csvfile = "D:\\Deep Learning\\Titanic\\Data\\test.csv"

df = pd.read_csv(csvfile, header=0)

#df.info()

cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)
ori_df = df


dummies = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
    dummies.append(pd.get_dummies(ori_df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
#titanic_dummies.info()

df = pd.concat((ori_df, titanic_dummies), axis=1)
df = df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
#df.info()
df['Age'] = df['Age'].interpolate()
df['Fare'] = df['Fare'].interpolate()
df.info()

X_results = df.values
X_results = np.delete(X_results, 1, axis=1)


clf = ensemble.GradientBoostingClassifier()

y_results = clf.predict(X_results)
clf.fit(X_results, y_results)

output = np.column_stack((X_results[:, 0], y_results))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID','Survived'])
df_results.to_csv('titanic_results.csv', index=False)

#kaggle competitions submit -c titanic -f submission.csv -m "Message"