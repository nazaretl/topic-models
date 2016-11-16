import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


def generate_funct_cloud(size=100, x_range=[-10, 10], sigma=400):
    """Generate a cloud of data points with given size in the shape similar to a given function"""
    def funct(x):
        """Function that will be approximated"""
        return x ** 3

    def random_float(a, b):
        """Create a random float in the range [a,b]"""
        return (b - a) * np.random.random_sample() + a

    x_values = [random_float(x_range[0], x_range[1]) for i in range(size)]
    data = []
    for x in x_values:
        data.append([x, np.random.normal(funct(x), sigma, 1)[0]])
    df = pd.DataFrame(data, columns=["x", "y"])
    return df

df = generate_funct_cloud()


def decisiontree_through_cloud(df, n=100):
    """Draw regression lines using Decision Trees"""
    X = np.array(df['x'])[:, None]
    Y = np.array(df['y'])
    ax = df.plot(x="x", y="y", figsize=(100, 60), kind="scatter", label="data", title="DecisionTree")
    P = (np.array([j for j in range(-10,10) for i in range(n)]).reshape(2*10*n, 1))
    for m_d in [3, 10]:
        dtr = DecisionTreeRegressor(max_depth=m_d)
        dtr.fit(X, Y)
        dtr_pred = dtr.predict(P)
        d = pd.DataFrame(P, columns=["P"])
        d["pred"] = dtr_pred
        d.sort_values("P").plot(x="P", y="pred", kind="line", label="Depth %d" % m_d, ax=ax)
    plt.show()

decisiontree_through_cloud(df)