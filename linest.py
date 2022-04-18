import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([29,59,119,238,464,659]).reshape(-1,1)
y = np.array([0.004,0.009,0.027,0.027,0.051,0.165])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)


