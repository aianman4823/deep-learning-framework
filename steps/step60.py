if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezero import dataloaders
from dezero.core import Variable
import dezero
from dezero import optimizers
import numpy as np
from dezero.datasets import Spiral, MNIST, SinCurve
from dezero.dataloaders import DataLoader, SeqDataLoader
import dezero.functions as F
import dezero.layers as L
import dezero.models as M


# train_set = SinCurve(train=True)
# dataloader = SeqDataLoader(train_set, batch_size=3)
# x, t = next(dataloader)
# print(x)
# print('================')
# print(t)

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)


class BetterRNN(M.Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y


model = BetterRNN(hidden_size, out_size=1)
optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        if epoch == 0 and count == 0:
            model.plot(x, to_file='rnn.png')
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    avg_loss = float(loss.data) / count

    print("| epoch %d | loss %f" % (epoch + 1, avg_loss))
