import torch
import dgl


src = torch.Tensor([0,1,2,3,0,1,2,3,2]).to(torch.int32).cuda()
dst = torch.Tensor([3,4,5,6,8,6,2,3,6]).to(torch.int32).cuda()
uni = torch.ones(20).to(torch.int32).cuda()
# indptr=torch.Tensor([0,2,3,3,3,6,6,7]) # src_bound
# indices=torch.Tensor([0,2,2,2,3,4,3]) # dst

# 
"""
替换目标:
signn.torch_sample_hop(
    self.cacheData[0],self.cacheData[1],
    sampleIDs,seed_num,fan_num,
    out_src,out_dst,out_num)
// 返回采样的edges

------------------------------------------

signn.torch_graph_mapping(all_node,cacheGraph[0],cacheGraph[1] \
    ,cacheGraph[0],cacheGraph[1],unique,edgeNUM,uniqueNUM)

"""
# print(src.dtype)
# sg1,sg2 = dgl.sampling.sample_with_arrays(src,dst)
print(src)
print(dst)
print(uni)
sg1,sg2,sg3 = dgl.remappingNode(src,dst,uni)

print(sg1)
print(sg2)
print(sg3)