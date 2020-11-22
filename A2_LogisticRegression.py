import torch

torch.manual_seed(1)
X = torch.Tensor([[-1, 1, 2],[1, 1, 1]])
y = torch.Tensor([-1, 1, 1])
w = torch.Tensor([[0.1],[0.1]])
alpha = 1

for iter in range(100):
    tmp = torch.exp(torch.matmul(torch.transpose(w,0,1),X)*(-y))

    ##############################
    ## Use tmp to compute f and g. Instead of summing we average the result, i.e.,
    ## complete only inside torch.mean(...) and don't remove this function
    ## Dimensions: f (scalar); g (2)
    ##############################
    #torch.mean returns the mean value of all elements in the tensor
    #preferred over sum b/c it is more understandable. if the dataset is grades on an exam, it's easier to understand an average than it is a sum over all exams -- the scale is better defined. If you train based off sum you should still get something similar
    f = torch.mean(torch.log(1+tmp))
    g = torch.mean((-y*tmp/(1+tmp))*X,1)
    print("Loss: %f; ||g||: %f" % (f, torch.norm(g)))
    # view(-1,1) keeps the same data but reshapes
    # for example could have t = torch.rand(4,4)
    # then b = t.view(2,8) transforms this matrix
    # -1 is a a wildcard; the columns are fixed to length of 1 and row takes whatevers left
    g = g.view(-1,1)
    w = w - alpha*g

print(w)
