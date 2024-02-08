"""对反向传播的自动微分进行单元测试"""

import sys
sys.path.append("repo/pydl")

from src.autodiff import *
import numpy as np

def test_backward():
    """利用数值微分进行梯度检验"""
    x = Variable(np.random.rand(1))  # 0~1随机输入

    def f(x):
        return square(exp(square(x)))

    # 反向传播前需要先进行前向传播
    y = f(x)

    # 反向传播
    y.backward()

    # 梯度检验
    x_grad = numerical_dirr(f, x)
    assert np.allclose(x.grad, x_grad, rtol=1e-05, atol=1e-08)  # 若x.grad和x_grad相近，返回True