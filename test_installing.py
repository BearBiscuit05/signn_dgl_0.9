import torch
import dgl


# src = torch.Tensor([0,1,2,3,0,1,2,3,2]).to(torch.int32).cuda()
# dst = torch.Tensor([3,4,5,6,8,6,2,3,6]).to(torch.int32).cuda()
# uni = torch.ones(20).to(torch.int32).cuda()
# print(src)
# print(dst)
# print(uni)
# sg1,sg2,sg3 = dgl.remappingNode(src,dst,uni)

# print(sg1)
# print(sg2)
# print(sg3)

src = torch.Tensor([0,4,4,5,5,7,8,10]).to(torch.int32).cuda()
dst = torch.Tensor([2,10,5,6,7,8,9,10,12,14]).to(torch.int32).cuda()
uni = torch.Tensor([0,1]).to(torch.int32).cuda()
seed_num = 2
fanout = 5
outSRC = torch.ones(20).to(torch.int32).cuda()
outDST = torch.ones(20).to(torch.int32).cuda()
print(outSRC)
print(outDST)
NUM = dgl.sampling.sample_with_edge(src, dst, uni, seed_num,fanout, outSRC,outDST)
print(NUM)
print(outSRC)
print(outDST)
