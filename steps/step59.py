if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezero.core import Variable
import dezero
from dezero import optimizers
import numpy as np
from dezero.datasets import Spiral, MNIST, SinCurve
from dezero.dataloaders import DataLoader
import dezero.functions as F
import dezero.layers as L
import dezero.models as M


max_epoch = 100
hidden_size = 100
bptt_length = 30

train_set = SinCurve(train=True)
seqlen = len(train_set)

model = M.SimpleRNN(hidden_size, 1)
optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))
