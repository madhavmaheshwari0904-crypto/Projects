import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'heart.csv')

data=pd.read_csv(csv_path)

"""print(data)
     Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  MaxHR ExerciseAngina  Oldpeak ST_Slope  HeartDisease
0     40   M           ATA        140          289          0     Normal    172              N      0.0       Up             0
1     49   F           NAP        160          180          0     Normal    156              N      1.0     Flat             1
2     37   M           ATA        130          283          0         ST     98              N      0.0       Up             0
3     48   F           ASY        138          214          0     Normal    108              Y      1.5     Flat             1
4     54   M           NAP        150          195          0     Normal    122              N      0.0       Up             0
..   ...  ..           ...        ...          ...        ...        ...    ...            ...      ...      ...           ...
913   45   M            TA        110          264          0     Normal    132              N      1.2     Flat             1
914   68   M           ASY        144          193          1     Normal    141              N      3.4     Flat             1
915   57   M           ASY        130          131          0     Normal    115              Y      1.2     Flat             1
916   57   F           ATA        130          236          0        LVH    174              N      0.0     Flat             1
917   38   M           NAP        138          175          0     Normal    173              N      0.0       Up             0

[918 rows x 12 columns]
"""


#EDA

"""print(data.columns)
Index(['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
       'HeartDisease'],
      dtype='str')
      """
#print(data.shape). (918, 12)
"""print(data.describe())
             Age   RestingBP  Cholesterol   FastingBS       MaxHR     Oldpeak  HeartDisease
count  918.000000  918.000000   918.000000  918.000000  918.000000  918.000000    918.000000
mean    53.510893  132.396514   198.799564    0.233115  136.809368    0.887364      0.553377
std      9.432617   18.514154   109.384145    0.423046   25.460334    1.066570      0.497414
min     28.000000    0.000000     0.000000    0.000000   60.000000   -2.600000      0.000000
25%     47.000000  120.000000   173.250000    0.000000  120.000000    0.000000      0.000000
50%     54.000000  130.000000   223.000000    0.000000  138.000000    0.600000      1.000000
75%     60.000000  140.000000   267.000000    0.000000  156.000000    1.500000      1.000000
max     77.000000  200.000000   603.000000    1.000000  202.000000    6.200000      1.000000
<class 'pandas.DataFrame'>"""


"""print(data.info())
RangeIndex: 918 entries, 0 to 917
Data columns (total 12 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Age             918 non-null    int64  
 1   Sex             918 non-null    str    
 2   ChestPainType   918 non-null    str    
 3   RestingBP       918 non-null    int64  
 4   Cholesterol     918 non-null    int64  
 5   FastingBS       918 non-null    int64  
 6   RestingECG      918 non-null    str    
 7   MaxHR           918 non-null    int64  
 8   ExerciseAngina  918 non-null    str    
 9   Oldpeak         918 non-null    float64
 10  ST_Slope        918 non-null    str    
 11  HeartDisease    918 non-null    int64  
dtypes: float64(1), int64(6), str(5)
memory usage: 86.2 KB
None
"""

#print(data.duplicated().sum()) 0
"""print(data["HeartDisease"].value_counts())
HeartDisease
1    508
0    410
1-Yes
0-No
"""

"""print(data.isnull().sum())
Age               0
Sex               0
ChestPainType     0
RestingBP         0
Cholesterol       0
FastingBS         0
RestingECG        0
MaxHR             0
ExerciseAngina    0
Oldpeak           0
ST_Slope          0
HeartDisease      0
dtype: int64
"""
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(data[var],kde=True)

"""plotting("Age",1)
plotting("RestingBP",2)
plotting("Cholesterol",3)
plotting("MaxHR",4)
plt.tight_layout()
plt.show()"""

#RestBP can't be 0, so we will replace it with mean value of RestBP
#Cholesterol can't be 0, so we will replace it with mean value of Cholesterol
print("Zeros before replacement - RestingBP:", (data["RestingBP"] == 0).sum(), "Cholesterol:", (data["Cholesterol"] == 0).sum())

rest = data.loc[data["RestingBP"] != 0, "RestingBP"].mean()
# Replace zeros in RestingBP with the mean computed from non-zero values
data["RestingBP"] = data["RestingBP"].replace(0, rest)

ch = data.loc[data["Cholesterol"] != 0, "Cholesterol"].mean()
# Replace zeros in Cholesterol with the mean computed from non-zero values
data["Cholesterol"] = data["Cholesterol"].replace(0, ch)

print("Zeros after replacement - RestingBP:", (data["RestingBP"] == 0).sum(), "Cholesterol:", (data["Cholesterol"] == 0).sum())



"""plotting("Age",1)
plotting("RestingBP",2)
plotting("Cholesterol",3)
plotting("MaxHR",4)
plt.tight_layout()
plt.show()"""

"""sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.show()
"""
#Data perprocessing and cleaning

df=pd.get_dummies(data,drop_first=True)
df=df.astype(int)
"""print(df)
     Age  RestingBP  Cholesterol  ...  ExerciseAngina_Y  ST_Slope_Flat  ST_Slope_Up
0     40        140          289  ...                 0              0            1
1     49        160          180  ...                 0              1            0
2     37        130          283  ...                 0              0            1
3     48        138          214  ...                 1              1            0
4     54        150          195  ...                 0              0            1
..   ...        ...          ...  ...               ...            ...          ...
913   45        110          264  ...                 0              1            0
914   68        144          193  ...                 0              1            0
915   57        130          131  ...                 1              1            0
916   57        130          236  ...                 0              1            0
917   38        138          175  ...                 0              0            1

[918 rows x 16 columns]
"""
from sklearn.preprocessing import StandardScaler
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
"""print(df.head())
        Age  RestingBP  Cholesterol  ...  ExerciseAngina_Y  ST_Slope_Flat  ST_Slope_Up
0 -1.433140   0.414885     0.834754  ...                 0              0            1
1 -0.478484   1.527224    -1.210675  ...                 0              1            0
2 -1.751359  -0.141284     0.722161  ...                 0              0            1
3 -0.584556   0.303651    -0.572651  ...                 1              1            0
4  0.051881   0.971054    -0.929194  ...                 0              0            1

[5 rows x 16 columns]
"""


################################
##### MODLE #################


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
X=df.drop("HeartDisease",axis=1)
y=df["HeartDisease"]
#print(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


models={
    "Logistic Regression": LogisticRegression(),
     "Decision Tree": DecisionTreeClassifier(),
     "Naive Bayes":GaussianNB(),
     "KNN":KNeighborsClassifier(),
     "SVM":SVC()
}
result=[]
for name,model in models.items():
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    result.append({
        "model":name,
        "Accuracy":round(acc,4),
        "F1_SCORE":round(f1,4)
    })
#print(result)
import joblib
joblib.dump(models["Logistic Regression"], "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.to_list(), "feature_columns.pkl")

