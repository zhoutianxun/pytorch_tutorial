import torch

# creating a tensor
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(my_tensor)

# creating a tensor with dtype and device
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
print(my_tensor)

# setting device if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# other common initialization methods
x = torch.empty(size=(3, 3))
print(x)
x = torch.zeros((3, 3))
print(x)
x = torch.rand((3, 3))
print(x)
x = torch.ones((3, 3))
print(x)
x = torch.eye(3, 3) # identity matrix
print(x)
x = torch.arange(start=0, end=5, step=1) # same as numpy
print(x)
x = torch.linspace(start=0.1, end=1, steps=10) # 10 evenly spaced between start and end
print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1) # create a normal distribution
print(x)
x = torch.diag(torch.ones(3))
print(x)

# Creating tensors of different types and converting to different types
tensor = torch.arange(4)
print(tensor.bool()) # boolean
print(tensor.short()) # int16
print(tensor.long()) # int64
print(tensor.half()) # float16
print(tensor.float()) # float32
print(tensor.double()) # float64

# Array to tensor and vice-versa
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array = tensor.numpy()
print(type(tensor))
print(type(np_array))

# Tensor math and comparion operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])
print(x + y)
print(x - y)
print(x / y)

# Inplace operations
t = torch.zeros(3)
t.add_(x) # underscore -> inplace operation

# Exponentiation
z = x.pow(2) # element wise power of 2
z = x**2

# Comparions
z = x > 0
print(z)

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x3)

# Matrix exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3) # matrix multiplied by itself by 3 times
print(matrix_exp)


# Element wise multiplication
z = x * y

# Dot product
z = torch.dot(x, y)

# Batch matrix multiplications
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out = torch.bmm(tensor1, tensor2)
print(out.shape) # shape will be (batch, n, p)

# Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2 # automatically expands the 1st dimension of x2 to match x1, and perform element-wise subtraction

# Other useful operations
sum_x1 = torch.sum(x1, dim=0) # sum up rows
print(sum_x1)
sum_x1 = torch.sum(x1, dim=1) # sum up columns
print(sum_x1)

values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_value = torch.abs(x)
mean_value = torch.mean(x.float(), dim=0)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10) # clamp x between min and max

x = torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool)
print(torch.any(x)) # check if any is true
print(torch.all(x)) # check if all are true

# Tensor indexing
batch_size = 10
features = 25
x = torch.rand(batch_size, features)

print(x[0].shape) # first example of the batch
print(x[:,0].shape) # first column of the whole batch
print(x[2:5, 0:10].shape)

x = torch.arange(10)
indices = [2, 5, 6]
print(x[indices]) # slicing with indices

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 1])
print(x[rows, cols].shape)

# indexing with condition
x = torch.arange(10)
print(x[(x < 3) | (x > 8)])
print(x[x%2 == 0])
print(torch.where(x > 5, x, x*2))

# other useful operations
print(x.numel()) # number of elements
print(x.ndim) # number of dimensions

# Reshaping tensors
x = torch.arange(9)
print(x.view(3, 3))
print(x.reshape(3, 3))
x = x.view(3, 3)

# transpose
print(x.t())
print(x.T)
print(x.transpose(1, 0))

# concatenate tensors
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0)) # cat on first dimension
print(torch.cat((x1, x2), dim=1)) # cat on second dimension

# Manipulating dimensions
batch = 64
x = torch.rand((batch, 2, 5))
print(x.view(-1)) # flatten matrix
print(x.view(batch, -1)) # keep number of batches, flatten by batch

z = x.permute(0, 2, 1) # swap dimensions
print(z.shape)

x = torch.arange(10) # shape (10,)
print(x.unsqueeze(0).shape) # expands the dimension at the 0th dim, shape -> (1, 10)

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # shape (1, 1, 10)
print(x.shape)
x = x.squeeze(0) # squeeze / collapse the 0th dimension, shape (1, 10)
print(x.shape)
x = x.squeeze(0) # shape (10,)