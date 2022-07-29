
from numpy import gradient, require
import torch
from torch.autograd import Variable

print('--Gaurav-- 29-07-2022- 15.27--')
# create a variable
x = Variable(torch.ones(2, 2), requires_grad = True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

# more operation on y

z = y * y * 3
out = z.mean()

print(z, out)

# Gradients
"""
let's backprop now out.backward() is equivalent to doing
out.backward(torch.Tensor([1.0]))
"""
out.backward()
# print gradient d(out)/dx
print(x.grad)


print('---new cases---')
x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)