from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

data = load_iris()
train = data.data
target = data.target
target = target.reshape(-1,1)
enc.fit(target)
target = enc.transform(target).toarray()