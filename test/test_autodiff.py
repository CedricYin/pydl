# 如果要运行该文件，需要先export PYTHONPATH=${PYTHONPATH}:/path/to/pydl，避免src模块找不到

from src.autodiff import *
import numpy as np

def test_backward():
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))

    # 反向传播前需要先进行前向传播，并存储中间值，这里是输出
    a = A(x)
    b = B(a)
    y = C(b)

    # 反向传播
    y.grad = np.array(1.0)
    y.backward()
    
    assert x.grad == 3.297442541400256