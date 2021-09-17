if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Parameter, Model
import dezero.layers as L
import dezero.functions as F
from dezero import optimizers, models


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iters = 10000
hidden_size = 10

model = models.MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(target=model)

for i in range(max_iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.updata()
    if i % 1000 == 0:
        print(loss)
