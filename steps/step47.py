if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.core import as_variable
import numpy as np
from dezero import Variable, Parameter, Model
import dezero.layers as L
import dezero.functions as F
from dezero import optimizers, models


# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# # get_item関数はnumpyのスライス操作にあたる関数
# # y = x[:, 2]
# y = F.get_item(x, [(0, 1), (1, 2)])
# print(y)
# y.backward()
# print(x.grad)

# def softmax1d(x):
#     x = as_variable(x)
#     y = F.exp(x)
#     sum_y = F.sum(y)
#     return y / sum_y


model = models.MLP((10, 3))
# x = np.array([[0.2, -0.4]])
# y = model(x)
# # p = F.softmax(y)
# p = softmax1d(x)
# p.backward()


# print(y)
# print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])

y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)

print(loss)
