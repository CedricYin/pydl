"""对算子进行单元测试"""

from src.autodiff import *
import numpy as np

def test_square():
    x = Variable(np.array(2.0))
    y = square(x)
    assert y.data == np.array(4.0)

def test_exp():
    x = Variable(np.array(0))
    y = exp(x)
    assert y.data == np.array(1)