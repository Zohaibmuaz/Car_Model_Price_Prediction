import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
data = pd.read_csv("data.csv")
data.head()
data.isnull().sum()
data.info()
data.CarName.unique()
data.price
data = data.select_dtypes(exclude=["object"])
data.head()
x = data.drop(["price"],axis=1)
y = data.price
x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.1,random_state=42)
model = DecisionTreeRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_valid)
mae = mean_absolute_error(y_valid,y_pred)
mae
model.score(x_valid,y_pred)
results =  pd.DataFrame({"Actual " : y_valid, "Predicted" : y_pred})
results.head()
results["Difference"] = y_valid - y_pred
results
