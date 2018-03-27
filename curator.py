import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_FILE_PATH = 'data/train.csv'


# load data
def loadData(filepath=DATA_FILE_PATH):
    return pd.read_csv(filepath)

title_map = {'Sir' : ('Capt', 'Don', 'Major', 'Sir'), \
              'Lady' : ('Dona', 'Lady', 'the Countess', 'Jonkheer', 'Mme', 'Mlle') , 'Miss' : ('Miss', 'Ms')}  


def findTitle(name):
    title = name.split(",")[1].strip().split(".")[0]
    for uniqTitle, titles in title_map.items():
        if title in titles:
            return uniqTitle
    return title

def ageImputer(data):
    parch = data["Parch"]; sibsp = data["SibSp"]
    if data["Age"] >= 0:
        return data["Age"]    
    if (parch == 0 and sibsp == 0):
        return 25
    elif (parch == 0 and sibsp == 1) or (parch >= 1 and sibsp >= 1):
        return 40
    elif parch >= 1 and parch < 3:
        return 9
    else:
        return 56
    
def preProcessAndGet(filepath):
    titanic_data = loadData(filepath)
    titanic_data["title"] = titanic_data["Name"].apply(findTitle)
    titanic_data["Family_Count"] = titanic_data["SibSp"] + titanic_data["Parch"]
    drop_columns = ["Cabin", "Name", "Ticket"]
    for col in drop_columns:
        titanic_data = titanic_data.drop(col, axis=1)
    
     
    titanic_data["Age"].fillna(-1, inplace=True)
    tit_age = titanic_data.apply(lambda x : ageImputer(x), axis=1)
    titanic_data["Age"] = tit_age
    titanic_data["Embarked"].fillna("S", inplace=True)
    
    # Label encoding male/female and embarkment
#     encoder = LabelEncoder()
#     embarked = titanic_data["Embarked"]
#     titanic_data["Embarked"] = encoder.fit_transform(embarked)
#     
#     encoder = LabelEncoder()
#     sex = titanic_data["Sex"]
#     titanic_data["Sex"] = encoder.fit_transform(sex)
    return titanic_data

