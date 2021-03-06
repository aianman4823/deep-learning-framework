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


# x = np.array([1, 2, 3])
# np.save('test.npy', x)

# x = np.load('test.npy')
# print(x)

max_epoch = 3
batch_size = 100
lr = 0.1

train_set = MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = M.MLP((1000, 10))
optimizer = optimizers.SGD(lr=lr).setup(model)

# パラメータの読み込み
if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        # 1バッチサイズ当たりのロス
        sum_loss += float(loss.data) * len(t)

    print("epoch: {}, loss: {:.4f}".format(
        # 全てのバッチサイズあたりのロス
        epoch + 1, sum_loss / len(train_set)
    ))

model.save_weights('my_mlp.npz')
