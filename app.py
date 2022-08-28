from numpy import float16
from flask import Flask, render_template
from flask import request
import pandas as pd

path = 'Fish.csv'

df = pd.read_csv(path)

df = df.dropna()
df.info()

X= df.drop('Species', axis = 1)
y= df['Species']

from sklearn.model_selection import KFold, cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.30, random_state=42)

#do not show warnings
import warnings
warnings.filterwarnings("ignore")

#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

#create an array of models
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))

#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=2)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result)

    xgb_model = xgb.XGBClassifier().fit(X_train, y_train)

import joblib

joblib.dump(xgb_model, "xgb.pkl") #export ML model to pkl file

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])

def text():
  if request.method == 'POST':
    
    xgb = joblib.load("xgb.pkl")
    # Get values through input bars
    Weight = request.form.get("Weight")
    Length1 = request.form.get("Length1")
    Length2 = request.form.get("Length2")
    Length3 = request.form.get("Length3")
    Height = request.form.get("Height")
    Width = request.form.get("Width")

    # Put inputs to dataframe
    X = pd.DataFrame([[Weight,Length1,Length2,Length3,Height,Width]], 
                     columns = ["Weight","Length1","Length2","Length3","Height","Width"])
    X = X.astype(float16)

    # Get prediction
    predict = xgb.predict(X)[0]

    if predict == 1.0:
      prediction = "Bream"
    
    elif predict == 2.0:
      prediction = "Roach"

    elif predict == 3.0:
      prediction = "Whitefish"

    elif predict == 4.0:
      prediction = "Parkki"

    elif predict == 5.0:
      prediction = "Perch" 

    elif predict == 6.0:
      prediction = "Pike "

    elif predict == 7.0:
      prediction = "Smelt"     

    else:
      prediction = "Error"

  else:
    prediction = 'Unknown'

  return render_template('text.html', output = prediction)

if __name__ == "__main__":
	app.run(debug=True)