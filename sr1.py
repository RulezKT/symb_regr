import json

from pysr import PySRRegressor
import numpy as np

# dt = np.dtype(np.float64)

with open("sun_lng_86400_pos_53693el.json", "r") as file:
    data = json.load(file)
    numpy_array = np.array(data)

# data_frame = pd.read_json('equin_solst_flat_grad.json')
# numpy_array = np.fromfile("equin_solst_flat_grad.json", sep=",")
# print(numpy_array)
# print(numpy_array.shape)

new_array = []
grads_array = []
for idx, _ in enumerate(numpy_array):
    if idx == 0:
        continue
    # grads_for_1_sec = (numpy_array[idx][1] - numpy_array[idx - 1][1]) / 86400
    grads_for_1_day = numpy_array[idx][1] - numpy_array[idx - 1][1]
    new_array.append(
        [idx - 1, grads_for_1_day],
    )
    grads_array.append(numpy_array[idx][1] / 100000)

# print(len(new_array))
# print(new_array[0:10])

# numpy_array = np.array(new_array)


X = np.array(new_array)  # numpy_array[:, 0]
# X[0] = X[0].astype(np.int64)
# X = X.reshape(-1, 1)


y = np.array(grads_array)  # numpy_array[:, 1]

# print(X)
# print(y)


# X = 2 * np.random.randn(100, 5)
# y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5


# Define the model
model = PySRRegressor(
    niterations=200_000,
    binary_operators=["+", "*", "-", "/", "mod"],
    unary_operators=["sin", "cos", "square", "cube", "abs", "sqrt"],
    model_selection="accuracy",
    batching=True,
    batch_size=100,
    ncycles_per_iteration=1100,  # default = 550
)

model.fit(X, y)

print(model)


# 12,3.2223838e-11,"9.855886e-6 * (x0 + (cos(x0 * (-0.046438966 * 0.37014103)) * 2.1076875))" sun / 100_000
# 12,2.1190405e-11,"(((cos(x0 * -0.017191013) / 0.63306254) + x0) * 9.856299e-6) + -2.3877637e-5"
