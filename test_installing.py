import torch
import dgl



"""
dgl.remappingNode(src,dst,uni)
    用于对src和dst进行ID重排,使得满足训练要求
    
    args:
        src(int32,cuda) : 边的源节点列表
        dst(int32,cuda) : 边的终点列表
        uni(int32,cuda) : 长度为src/dst的2倍,用于接收返回的uni节点列表
    
    return:
        new_src(int32,cuda) : 重排后的源节点id
        new_dst(int32,cuda) : 重排后的终点id
        uniId(int32,cuda)   : 重排的id对应 index为new id,对应位置为原来的id

"""
# print("===> Testing remapping func")
# src = torch.Tensor([11,12,13,17,14,15,16,17,17]).to(torch.int32).cuda()
# dst = torch.Tensor([10,10,10,10,11,11,12,12,13]).to(torch.int32).cuda()
# uni = torch.ones(20).to(torch.int32).cuda()
# print("raw src:",src)
# print("raw dst:",dst)
# print("raw uni:",uni)

# sg1,sg2,sg3 = dgl.remappingNode(src,dst,uni)
# print(uni)
# print("mapped src:",sg1)
# print("mapped dst:",sg2)
# print("mapped uni:",sg3)


"""
dgl.sampling.sample_with_edge(inptr, indices, seed, seed_num,fanout, outSRC,outDST)
    对训练节点进行一跳采样
    
    args:
        inptr(int32,cuda)   : 图的inptr,一般为dst
        indices(int32,cuda) : 图的indices,一般为src列
        seed(int32,cuda)    : 采样列表
        seed_num(int)       : 采样节点数目
        fanout(int)         : 采样邻居数目
        outSRC(int32,cuda)  : 采样后得到的src列,COO结构
        outDST(int32,cuda)  : 采样后得到的dst列,COO结构
    return:
        NUM(int32)          : COO结构的长度
"""
# src = torch.Tensor([0,4,5,7,8,10]).to(torch.int32).cuda()
# dst = torch.Tensor([2,10,5,6,7,8,9,10,12,14]).to(torch.int32).cuda()
# uni = torch.Tensor([0,2]).to(torch.int32).cuda()
# seed_num = 2
# fanout = 5
# outSRC = torch.zeros(20).to(torch.int32).cuda()
# outDST = torch.zeros(20).to(torch.int32).cuda()
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


"""
dgl.fastFindNeighbor(nodeTable,src,dst,accumulate=False,flag=1)
    用于使用cuda进行快速BFS的函数
    
    args:
        nodeTable(int32,cuda)   : 用于记录节点是否被访问的向量
        src(int32,cuda)         : 表示图的起始列
        dst(int32,cuda)         : 表示图的终点列
        accumulate(bool)        : 表示是否需要对重复的访问进行累加
        flag(int64)             : 表示访问的标记内容
    
    return:
        None: 仅在nodeTable中进行修改
"""
# nodeTable = torch.Tensor([0,0,1,0,0,0,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([0,2,4,5,2,4,5,2]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,3,3]).to(torch.int32).cuda()
# acc = False
# print("===> BFS func test")
# print(nodeTable)
# dgl.fastFindNeighbor(nodeTable,src,dst,acc)
# print(nodeTable)



"""
dgl.fastFindNeigEdge(nodeTable,edgeTable,src_batch,dst_batch,offset)
    使用BFS检测边被遍历的情况

    args:
        nodeTable(int32,cuda) : 用于记录节点是否被访问的向量
        edgeTable(int32,cuda) : 用于记录边是否被访问的向量
        src_batch(int32,cuda) : 传入图的部分起始边,减少内存压力
        dst_batch(int32,cuda) : 传入图的部分终点边,减少内存压力
        offset(int)           : 传入当前部分边的偏移量
    
    return:
        None: 仅在nodeTable,edgeTable中进行修改
"""
# nodeTable = torch.Tensor([0,0,1,0,0,0,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([0,2,4,5,2,4,5,2]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,3,1]).to(torch.int32).cuda()
# src_batches = torch.chunk(src, 3, dim=0)
# dst_batches = torch.chunk(dst, 3, dim=0)
# batch = [src_batches, dst_batches]
# offset = 0
# edgeTable = torch.zeros(len(dst),dtype=torch.int32).cuda()
# print("===> BFS edge func test")
# print(edgeTable)

# for src_batch,dst_batch in zip(*batch):
#     print("src_batch:",src_batch)
#     print("dst_batch:",dst_batch)
#     dgl.fastFindNeigEdge(nodeTable,edgeTable,src_batch,dst_batch,offset)
#     print("nodeTable:",nodeTable)
#     print("edgeTable:",edgeTable)
#     offset += len(src_batch)
#     print("offset:",offset)
#     print('-'*10)
# print(f"nodeTable :{nodeTable}")
# print(f"edgeTable :{edgeTable}")


"""
dgl.mapLocalId(mapTable,Gid,Lid)
    用于通过映射表快速查找映射后的结果    

    args:
        mapTable(int32,cuda): 表示记录map的id对应关系
        Gid(int32,cuda)     : 表示全局id
        Lid(int32,cuda)     : 存储map后的局部id结果
    
    return:
        None: Lid中进行修改
"""
# nodeTable = torch.Tensor([0,5,4,7,2,3,1,8]).to(torch.int32).cuda()
# Gid = torch.Tensor([0,1,2]).to(torch.int32).cuda()
# Lid = torch.zeros_like(Gid).to(torch.int32).cuda()
# dgl.mapLocalId(nodeTable,Gid,Lid)
# print(Lid)



"""
dgl.sumDegree(nodeTable,src,dst)
    用于快速计算每个节点的入度情况

    args:
        nodeTable(int32,cuda)   : 记录每个点的入度情况
        src(int32,cuda)         : 表示图的起始列
        dst(int32,cuda)         : 表示图的终止列
    
    return:
        None: nodeTable中进行修改
"""
# nodeTable = torch.Tensor([0,0,0,0,0,0,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([0,2,4,5,2,4,5,2]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,3,1]).to(torch.int32).cuda()
# nodeTable = dgl.sumDegree(nodeTable,src,dst)


# PTable = torch.Tensor([0,0,1000,1000,1000,1000]).to(torch.int32).cuda()
# DegreeTable = torch.Tensor([3,2,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([2,3,4,4,5]).to(torch.int32).cuda()
# dst = torch.Tensor([0,0,0,1,1]).to(torch.int32).cuda()
# fanout = 1
# print(PTable)
# PTabel = dgl.calculateP(DegreeTable,PTable,src,dst,fanout)
# print(PTabel)


"""
dgl.pagerank(src,dst,degreeTable,nodeValue,nodeInfo)
    用于计算每个点的pagerank,并传递分区信息

    args:
        src(int32,cuda)         : 表示图的起始列
        dst(int32,cuda)         : 表示图的终止列
        degreeTable(int32,cuda) : 表示图度信息
        nodeValue(int32,cuda)   : 表示图权值
        nodeInfo(int32,cuda)    : 表示图需要传递的信息
    
    return:
        None: 直接在nodeValue,nodeInfo中进行修改
"""
# degreeTable = torch.Tensor([1,0,3,0,2,2,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([0,2,4,5,2,4,5,2]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,3,1]).to(torch.int32).cuda()
# nodeValue = torch.Tensor([0,0,10000,0,0,0,0,0]).to(torch.int32).cuda()
# nodeInfo = torch.Tensor([0,0,2,0,0,0,0,0]).to(torch.int32).cuda()
# dgl.per_pagerank(src,dst,degreeTable,nodeValue,nodeInfo)
# print("nodeValue:",nodeValue)
# print("nodeInfo:",nodeInfo)

"""
分区中
首先抽取对应id
随后抽取重要度
然后按照重要度进行重排
对重排结果分成不同的格内部
"""