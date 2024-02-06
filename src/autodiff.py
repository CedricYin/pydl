"""基于define-by-run（动态计算图），实现自动微分"""

import numpy as np


class Variable:
    """变量类，对实际数据进行封装"""
    def __init__(self, data):
        # 只支持numpy.ndarray类型的数据
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.grad = None
        self.creator = None  # 创造该变量的函数
    
    def set_creator(self, func):
        self.creator = func

    # 递归方式实现反向传播  
    # def backward(self):
    #     f = self.creator
    #     if f is not None:
    #         x = f.input
    #         x.grad = f.backward(self.grad)
    #         x.backward()  # 递归地反向传播到最左边的参数（最左边的参数的creator为None
    
    # 循环方式实现反向传播
    def backward(self):
        # 最终的输出对自己的导数是1
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output  # 获取中间结果
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    """所有计算函数（算子）的基类"""
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)  # define-by-run, 在计算之间建立连接
        # 保存中间数据
        self.input = input  # 保存输入变量（反向传播所需要的中间值，这里是输入）
        self.output = output  # 保存输出变量（反向传播所需要的中间值，这里是输出）
        return output
    
    def forward(self, x):
        """根据计算图，前向传播计算"""
        raise NotImplementedError
    
    def backward(self, gy):
        """根据计算图，并基于链式法则，反向传播求导"""
        raise NotImplementedError


class Square(Function):
    """计算平方"""
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


def square(x):
    """暴露给用户的接口：计算平方"""
    return Square()(x)
    

class Exp(Function):
    """计算exp"""
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    """暴露给用户的接口：计算exp"""
    return Exp()(x)


def as_array(x):
    """将x转化为ndarray：
    因为0维的ndarray计算后会得到标量类型，这里为了避免TypeError，实现了该函数"""
    if np.isscalar(x):
        return np.array(x)
    return x


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
    x = np.array(2)
    print(x ** 2)
    print(type(x ** 2))
