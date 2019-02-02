import torch.cuda

cuda = torch.device('cuda')

y = torch.Tensor([1., 2.]).cuda()
print(y)
print(y.device)
