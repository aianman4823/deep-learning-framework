if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import dezero
from dezero import optimizers
import numpy as np
from dezero.datasets import Spiral, MNIST
from dezero.dataloaders import DataLoader
import dezero.functions as F
import dezero.models as M

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with dezero.test_mode():
    y = F.dropout(x)
    print(y)
