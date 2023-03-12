import paddle
import torch
from fastcore.all import patch_to, partial # 1.0
# def create_tensor():
#     a =paddle.randn([3,4])
#     print(a.long())
#     return a
#
#
# def repeat(self,*sizes):
#     print(sizes)
#
# repeat(1,2,3)

def torch_zeros(*size,
            out=None,
            dtype=None):
    return paddle.zeros(shape=size, dtype=dtype) # 递归调用自己修改过的

paddle.torch_zeros = torch_zeros # 修改为自定义

import time
t1= time.time()
r = 100000
for _ in range(r):
    # x = paddle.torch_zeros(3, 4, 5) # 5.7
    x = paddle.zeros([3, 4, 5]) # 5.736
t2 = time.time()
print((t2-t1))
print(paddle.empty([3]))
print(torch.empty(3))