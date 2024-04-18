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

# Sun
# each x = 86400 seconds
# 10, 1.4166603, "((x0 * 0.98560095) + -1.201872) + sin(-0.017168723 * x0)"
# eph_time = 0
# hz = 0.98560226 * (sin(0.017194912365632 * eph_time + 1.3488353) + eph_time - 1.1195226)
# 14,0.7946078,"0.98560226 * (sin(((x0 * -0.02196941) * -0.7826752) - -1.3488353) + (x0 + -1.1195226))"


# Moon
