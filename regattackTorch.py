import torch
import torch.nn.functional as F

# fast gradient sign
def FGSM(model, criterion, x, y, epsilon=1e-3):
    x = torch.autograd.Variable(x, requires_grad=True)
    prediction = model(x)
    loss = criterion(prediction, y)
    grad = torch.autograd.grad(loss, x)[0]
    return x + torch.mul(torch.sign(grad), epsilon)

# iterative fast gradient sign
def IFGSM(model, criterion, x, y, epsilon=1e-3, epochs=20):
    x = torch.autograd.Variable(x, requires_grad=True)
    for i in range(epochs):
        prediction = model(x)
        loss = criterion(prediction, y)
        grad = torch.autograd.grad(loss, x)[0]
        x = x + torch.mul(torch.sign(grad), epsilon/epochs)
    return x

# i=j=2 attack from IJCAI workshop paper
def attack22(model, sigma, x):
    J = torch.autograd.functional.jacobian(model, x)
    mat = torch.matmul(J, torch.sqrt(sigma))
    U, S, V = torch.linalg.svd(mat)
    # find max eigenvalue
    max = 0
    maxidx = -1
    for i in range(len(S)):
        if S[i] > max:
            max = S[i]
            maxidx = i
    return F.normalize(V[:,maxidx], p=2, dim=0)

# i=j=1 attack from IJCAI workshop paper
def attack11(model, sigma, x):
    e = torch.zeros(len(x))
    J = torch.autograd.functional.jacobian(model, x)
    mat = torch.matmul(J, torch.sqrt(sigma))
    max = 0
    maxidx = -1
    for i in range(mat.shape[1]):
        n = torch.linalg.norm(mat[:,i], ord=1)
        if n > max:
            max = n
            maxidx = i
    e[i] = 1
    return e