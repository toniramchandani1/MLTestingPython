import pytest
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@pytest.fixture(scope='module')
def model_iris():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    return dtc, X_test, y_test

def test_accuracy(model):
    dtc, X_test, y_test = model
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.9

def test_shape(model):
    dtc, X_test, y_test = model
    assert X_test.shape[1] == 3
    assert y_test.shape[0] == 30

if __name__ == '__main__':
    pytest.main()