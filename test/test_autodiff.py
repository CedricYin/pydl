# 如果要运行该文件（不使用pytest），
# 需要先export PYTHONPATH=${PYTHONPATH}:/path/to/pydl，避免src模块找不到。

from src.autodiff import *
import numpy as np

def test_backward():
    x = Variable(np.array(0.5))

    # 反向传播前需要先进行前向传播
    a = square(x)
    b = exp(a)
    y = square(b)

    # 反向传播
    y.backward()

    assert x.grad == 3.297442541400256