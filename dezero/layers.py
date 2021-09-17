import numpy as np
from dezero.core import Variable, Parameter
import dezero.functions as F
import weakref


class Layer:
    def __init__(self):
        self._params = set()

    # __setattr__メソッドはインスタンス変数を設定するたびに呼び出される
    # つまり、 x = Layer()では__init__が呼び出され、その後
    # x.p1 = y, x.p2 = y2のように呼び出すと、nameにp1がvalueにyが格納される
    # nameにはinstance変数の名前が、vlaueにはインスタンス変数の値が格納される
    # また、__setattr__を利用してnameとvalueを格納すると__dict__メソッドを
    # 利用して、アクセスできる
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        # superを利用することで今すでに定義されているinstance変数を呼び出せる
        # つまり、layer = Layer()と最初に定義しているならそれ！
        # こうしないとlayer.p1のように新しいインスタンス変数で__init__メソッドないで
        # 呼び出された変数を保持できない
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, 'W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # データを流すタイミングでWを初期化
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y
