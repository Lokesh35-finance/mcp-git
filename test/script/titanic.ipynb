
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../data/train.csv")
df.head()
1	2	3	4	5	6	7	8	9	10
0	sn	pclass	survived	NaN	gender	age	family	fare	embarked	date
1	1	3	0	Mr. Anthony	male	42	0	7.55	NaN	01-Jan-90
2	1	3	0	Mr. Anthony	male	42	0	7.55	NaN	01-Jan-90
3	2	3	0	Master. Eugene Joseph	male	?	2	20.25	S	02-Jan-90
4	3	2	0	Abbott, Mr. Rossmore Edward	NaN	NaN	2	**	S	03-Jan-90
print("Shape of dataset:", df.shape)
print("\nData Types:\n", df.dtypes)
df.isnull().sum()
msno.matrix(df)
Shape of dataset: (1302, 10)

Data Types:
 1     object
2     object
3     object
4     object
5     object
6     object
7     object
8     object
9     object
10    object
dtype: object
<Axes: >

df.columns
Index(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], dtype='object')
df = pd.read_csv("../data/train.csv", header=0)
df.columns
Index(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], dtype='object')
raw_df = pd.read_csv("../data/train.csv", header=None)
raw_df.head()
0	1	2	3	4	5	6	7	8	9
0	1	2	3	4	5	6	7	8	9	10
1	sn	pclass	survived	NaN	gender	age	family	fare	embarked	date
2	1	3	0	Mr. Anthony	male	42	0	7.55	NaN	01-Jan-90
3	1	3	0	Mr. Anthony	male	42	0	7.55	NaN	01-Jan-90
4	2	3	0	Master. Eugene Joseph	male	?	2	20.25	S	02-Jan-90
print("Shape:", raw_df.shape)
Shape: (1303, 10)
import pandas as pd
df = pd.read_csv("../data/train.csv", skiprows=1)
df.head()
sn	pclass	survived	Unnamed: 3	gender	age	family	fare	embarked	date
0	1	3	0	Mr. Anthony	male	42	0.0	7.55	NaN	01-Jan-90
1	1	3	0	Mr. Anthony	male	42	0.0	7.55	NaN	01-Jan-90
2	2	3	0	Master. Eugene Joseph	male	?	2.0	20.25	S	02-Jan-90
3	3	2	0	Abbott, Mr. Rossmore Edward	NaN	NaN	2.0	**	S	03-Jan-90
4	4	3	1	Abbott, Mr. Rossmore Edward	female	35	2.0	20.25	S	04-Jan-90
df.columns = ['PassengerId', 'Pclass', 'Survived', 'Name', 'Sex', 
              'Age', 'Family', 'Fare', 'Embarked', 'Date']
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Mme', 'Lady', 'Dona'], 'Mrs')
df['Title'] = df['Title'].replace(['Capt', 'Don', 'Major', 'Sir', 'Col', 'Rev', 'Jonkheer', 'Dr'], 'Rare')
print(df['Title'].value_counts())
Title
Mr          749
Miss        262
Mrs         199
Master       60
Rare         26
Countess      1
Name: count, dtype: int64
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.countplot(x='Title', hue='Survived', data=df, palette='Set2')
plt.title("Survival Rate by Title")
plt.xlabel("Passenger Title")
plt.ylabel("Count")
plt.show()

df['FamilySize'] = df['Family'] + 1  
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df[['Family', 'FamilySize', 'IsAlone']].head()
Family	FamilySize	IsAlone
0	0.0	1.0	1
1	0.0	1.0	1
2	2.0	3.0	0
3	2.0	3.0	0
4	2.0	3.0	0
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
sns.countplot(x='FamilySize', hue='Survived', data=df, palette='pastel')
plt.title("Survival by Family Size")
plt.xlabel("Family Size (Including Passenger)")
plt.ylabel("Count")
plt.show()

sns.countplot(x='IsAlone', hue='Survived', data=df, palette='muted')
plt.title("Survival Rate: Alone vs With Family")
plt.xlabel("Is Alone (1 = Alone)")
plt.ylabel("Count")
plt.show()

df.isnull().sum()
PassengerId      0
Pclass           0
Survived         0
Name             0
Sex              1
Age            257
Family           2
Fare             2
Embarked         6
Date             0
Title            4
FamilySize       2
IsAlone          0
dtype: int64
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.dropna(thresh=6) 
import numpy as np
df['Age'] = df['Age'].replace('?', np.nan)
df['Age'] = pd.to_numeric(df['Age'])
df['Age'] = df['Age'].fillna(df['Age'].median())
print(df['Age'].isnull().sum())
print(df['Age'].dtype)           
0
float64
from sklearn.preprocessing import LabelEncoder

# Encode 'Sex'
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

# Encode 'Embarked'
le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# Encode 'Title'
le_title = LabelEncoder()
df['Title'] = le_title.fit_transform(df['Title'])
df[['Sex', 'Embarked', 'Title']].head()
Sex	Embarked	Title
0	1	2	6
1	1	2	6
2	1	2	6
3	2	2	3
4	0	2	3
y = df['Survived']
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked', 'Title']
X = df[features]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X[~X.applymap(lambda x: str(x).replace('.', '', 1).isdigit())]
Pclass	Sex	Age	Fare	FamilySize	IsAlone	Embarked	Title
0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	NaN	NaN	NaN	**	NaN	NaN	NaN	NaN
4	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
...	...	...	...	...	...	...	...	...
1296	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1297	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1298	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1299	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1300	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1301 rows × 8 columns

# Replace invalid values like '**' or '?' with NaN
X['Fare'] = pd.to_numeric(X['Fare'], errors='coerce')

# Fill missing with median
X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
X['Age'] = X['Age'].fillna(X['Age'].median())
import numpy as np

# Columns that must be numeric
numeric_columns = ['Age', 'Fare', 'FamilySize']

for col in numeric_columns:
    # Replace non-numeric entries (like '**', '?') with NaN
    X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill NaN values with median
    X[col] = X[col].fillna(X[col].median())
print(X.dtypes)
Pclass          int64
Sex             int64
Age           float64
Fare          float64
FamilySize    float64
IsAlone         int64
Embarked        int64
Title           int64
dtype: object
import numpy as np

# Clean all columns in X (convert to numeric where applicable)
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            # Try converting to numeric
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            pass

# Fill any remaining NaNs with median (for numeric columns only)
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        X[col] = X[col].fillna(X[col].median())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
LogisticRegression(max_iter=200)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("\nClassification Report:\n", classification_report(y_test, rf_preds))
Random Forest Accuracy: 0.7739463601532567

Classification Report:
               precision    recall  f1-score   support

           0       0.77      0.87      0.81       149
           1       0.78      0.65      0.71       112

    accuracy                           0.77       261
   macro avg       0.78      0.76      0.76       261
weighted avg       0.78      0.77      0.77       261

from xgboost import XGBClassifier

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict and evaluate
xgb_preds = xgb_model.predict(X_test)

print("🚀 XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("\nClassification Report:\n", classification_report(y_test, xgb_preds))
🚀 XGBoost Accuracy: 0.8007662835249042

Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.85      0.83       149
           1       0.79      0.73      0.76       112

    accuracy                           0.80       261
   macro avg       0.80      0.79      0.79       261
weighted avg       0.80      0.80      0.80       261

import os
os.makedirs("models", exist_ok=True)
import joblib
joblib.dump(model, "models/titanic_model.pkl")
['models/titanic_model.pkl']
import joblib
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/titanic_model.pkl")
['models/titanic_model.pkl']
