import torch
import dgl




# print("===> Testing remapping func")
src = torch.Tensor([0,1,2,3,0,1,2,3,2]).to(torch.int32).cuda()
dst = torch.Tensor([3,4,5,6,8,6,2,3,6]).to(torch.int32).cuda()
uni = torch.ones(20).to(torch.int32).cuda()
print("raw src:",src)
print("raw dst:",dst)
print("raw uni:",uni)
### src: int32 cuda 
### dst: int32 cuda
### uni: int32 cuda
sg1,sg2,sg3 = dgl.remappingNode(src,dst,uni)
print("mapped src:",sg1)
print("mapped dst:",sg2)
print("mapped uni:",sg3)

# =========================sampleTest===========================
# src = torch.Tensor([0,4,4,5,5,7,8,10]).to(torch.int32).cuda()
# dst = torch.Tensor([2,10,5,6,7,8,9,10,12,14]).to(torch.int32).cuda()
# uni = torch.Tensor([0,1]).to(torch.int32).cuda()
# seed_num = 2
# fanout = 5
# outSRC = torch.ones(20).to(torch.int32).cuda()
# outDST = torch.ones(20).to(torch.int32).cuda()
# print(outSRC)
# print(outDST)
# NUM = dgl.sampling.sample_with_edge(src, dst, uni, seed_num,fanout, outSRC,outDST)
# print(NUM)
# print(outSRC)
# print(outDST)

# =========================haloTest===========================
# ptr = torch.Tensor([0,1,4,6,8,10]).to(torch.int32).cuda()
# inlice = torch.Tensor([3,-1,-1,-1,8,6,-1,-1,6,10]).to(torch.int32).cuda()
# edge = torch.Tensor([99,97,96,92,81,110]).to(torch.int32).cuda()
# bound = torch.Tensor([0,3,4,6]).to(torch.int32).cuda()
# gap = 0
# dgl.loadGraphHalo(ptr,inlice,edge,bound,gap)

# print(ptr)
# print(inlice)

# =========================BFSTest===========================
# nodeTable = torch.Tensor([0,0,1,0,0,0,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([0,2,4,5,2,4,5,2]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,3,3]).to(torch.int32).cuda()
# acc = False
# print("===> BFS func test")
# print(nodeTable)
# dgl.fastFindNeighbor(nodeTable,src,dst,acc)
# print(nodeTable)

# =========================BFSEdgeTest===========================
# nodeTable = torch.Tensor([0,0,1,0,0,0,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([0,2,4,5,2,4,5,2]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,3,1]).to(torch.int32).cuda()
# edgeTable = torch.zeros(len(dst),dtype=torch.int32).cuda()
# print("===> BFS edge func test")
# print(edgeTable)
# dgl.fastFindNeigEdge(nodeTable,edgeTable,src,dst)
# print(edgeTable)

# =========================BFSNodeMap===========================
# nodeTable = torch.Tensor([0,5,4,7,2,3,1,8]).to(torch.int32).cuda()
# Gid = torch.Tensor([0,1,2]).to(torch.int32).cuda()
# Lid = torch.zeros_like(Gid).to(torch.int32).cuda()
# dgl.mapLocalId(nodeTable,Gid,Lid)
# print(Lid)