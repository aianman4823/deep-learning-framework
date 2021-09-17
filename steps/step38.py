if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.random.randn(1, 2, 3))
y = x.transpose((1, 0, 2))

y.backward()
print(x.grad)
print(y.shape)
