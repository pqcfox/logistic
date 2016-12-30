from logistic import Logistic
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_cancer_dataset():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
    log = Logistic().fit(X_train, y_train)
    y_pred = log.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.95
