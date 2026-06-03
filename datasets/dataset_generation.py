import numpy as np
import pandas as pd

e = np.random.standard_normal(100)
x = (np.random.random(100)*2 - 1)*10

pd.DataFrame(
    {
        "x": x,
        "y": 2*x + 5*e
    }
).to_csv("datasets/linear_model.csv", index=False)

pd.DataFrame(
    {
        "x": x,
        "y": np.exp(x/4.5) + e
    }
).to_csv("datasets/exp_model.csv", index=False)

pd.DataFrame(
    {
        "x": x,
        "y": np.sin(1.5*x) + 0.5*e
    }
).to_csv("datasets/sin_model.csv", index=False)