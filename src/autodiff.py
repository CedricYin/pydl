"""实现自动微分"""

import numpy as np


class Variable:
    """变量类，对实际数据进行封装"""
    def __init__(self, data):
        self.data = data


class Function:
    """所有计算函数（算子）的基类"""
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    """计算平方"""
    def forward(self, x):
        return x ** 2


class Exp(Function):
    """计算exp"""
    def forward(self, x):
        return np.exp(x)


def numerical_dirr(f, x, eps=1e-4):
    """数值微分（中心差分近似）"""
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def demo_f(x):
    """复合函数示例: y = (e^(x^2))^2"""
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


if __name__ == "__main__":
    x = Variable(np.array(0.5))
    dy = numerical_dirr(demo_f, x)
    print(dy)
