# Packages
import joblib
from sklearn.model_selection import train_test_split
from sklearn import datasets
import packages.model_trainer as mt
import pandas as pd


# 0.Path to data
#path_to_data = './data/iris.csv'
iris = datasets.load_iris()
prepared_data = pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})

# 1.Prepare the data
#prepared_data = pd.read_csv(path_to_data)
prepared_data = pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})

# 2.Create train - test split
X_train, X_test, y_train, y_test = train_test_split(prepared_data.drop(['species'], axis=1),
                                                prepared_data['species'],
                                                        test_size=0.2, 
                                                        random_state=777)

# 3.Run training
model = mt.run_model_training(X_train, X_test, y_train, y_test)

# 4.Save the trained model and vectorizer
joblib.dump(model, './models/model.pkl')
