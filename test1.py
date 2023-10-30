import dgl
import torch
import numpy

# nodeTable = torch.Tensor([0,0,0,0,0,0,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([0,2,4,5,2,4,5,2]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,3,1]).to(torch.int32).cuda()
# nodeTable = dgl.sumDegree(nodeTable,src,dst)

# print(nodeTable)


PTable = torch.Tensor([0,0,1000,1000,1000,1000]).to(torch.int32).cuda()
DegreeTable = torch.Tensor([3,2,0,0,0,0]).to(torch.int32).cuda()
src = torch.Tensor([2,3,4,4,5]).to(torch.int32).cuda()
dst = torch.Tensor([0,0,0,1,1]).to(torch.int32).cuda()
fanout = 1
print(PTable)
PTabel = dgl.calculateP(DegreeTable,PTable,src,dst,fanout)
print(PTabel)