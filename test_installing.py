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

# src = torch.Tensor([0,5,5,7,8,10]).to(torch.int32).cuda()
# dst = torch.Tensor([2,10,5,6,7,8,9,10,12,14]).to(torch.int32).cuda()
# uni = torch.Tensor([0,2]).to(torch.int32).cuda()
# mapTable = torch.zeros(15).to(torch.int32)
# mapTable[5] = -1
# mapTable[8] = -1
# mapTable[9] = -1
# mapTable[10] = -1
# seed_num = 2
# fanout = 5
# mapTable = mapTable.cuda()
# outSRC = torch.zeros(20).to(torch.int32).cuda()
# outDST = torch.zeros(20).to(torch.int32).cuda()

# print(outSRC)
# print(outDST)
# print(mapTable)
# NUM = dgl.sampling.sample_with_edge_and_map(src, dst, uni, seed_num,fanout, outSRC,outDST,mapTable)
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
# src = torch.Tensor([0,2,4,5,2,4,2,5]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,1,3]).to(torch.int32).cuda()
# src_batches = torch.chunk(src, 3, dim=0)
# dst_batches = torch.chunk(dst, 3, dim=0)
# batch = [src_batches, dst_batches]
# offset = 0
# edgeTable = torch.zeros(len(dst),dtype=torch.int32)
# print("===> BFS edge func test")
# print(edgeTable)

# for src_batch,dst_batch in zip(*batch):
#     print('-'*10)
#     print("src_batch:",src_batch)
#     print("dst_batch:",dst_batch)
#     tmp = edgeTable[offset:offset+len(src_batch)].cuda()
#     dgl.fastFindNeigEdge(nodeTable,tmp,src_batch,dst_batch)
#     edgeTable[offset:offset+len(src_batch)] = tmp[:len(src_batch)].cpu()
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
dgl.sumDegree(inNodeTable,outNodeTable,src,dst)
    用于快速计算每个节点的入度情况
    src -> dst

    args:
        inNodeTable(int32,cuda)   : 记录每个点的入度情况
        outNodeTable(int32,cuda)   : 记录每个点的入度情况
        src(int32,cuda)         : 表示图的起始列
        dst(int32,cuda)         : 表示图的终止列
    
    return:
        # TODO:
        None: nodeTable中进行修改
"""

# inNodeTable = torch.Tensor([0,0,0,0,0,0,0,0,0,0]).to(torch.int32).cuda()
# outNodeTable = torch.Tensor([0,0,0,0,0,0,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([0,2,4,5,2,4,5,2]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,3,1]).to(torch.int32).cuda()
# inNodeTable,outNodeTable = dgl.sumDegree(inNodeTable,outNodeTable,src,dst)
# print(inNodeTable)
# print(outNodeTable)

# PTable = torch.Tensor([0,0,1000,1000,1000,1000]).to(torch.int32).cuda()
# DegreeTable = torch.Tensor([3,2,0,0,0,0]).to(torch.int32).cuda()
# src = torch.Tensor([2,3,4,4,5]).to(torch.int32).cuda()
# dst = torch.Tensor([0,0,0,1,1]).to(torch.int32).cuda()
# fanout = 1
# print(PTable)
# PTabel = dgl.calculateP(DegreeTable,PTable,src,dst,fanout)
# print(PTabel)


"""
dgl.mapByNodeSet(nodeTable,uniTable,srcList,dstList)
    使用nodeTable进行id重排,随后对src和dst重新赋值

    args:
        nodeTable(int32,cuda)   : 出入需要排序的nodeTable
        uniTable(int32,cuda)    : 接收nodeTable重排后的id
        src(int32,cuda)         : 表示图的起始列
        dst(int32,cuda)         : 表示图的终止列
    
    return:
        src(int32,cuda)         : 表示重排后图的起始列
        dst(int32,cuda)         : 表示重排后图的终止列
        uniTable(int32,cuda)    : 图被排序后的id映射表,长度已进行修改
"""
# nodeTable = torch.Tensor([1,0,5,4,0,1,2,3]).to(torch.int32).cuda()
# uniTable = torch.zeros_like(nodeTable).to(torch.int32).cuda()
# Gid = torch.Tensor([2]).to(torch.int32).cuda()
# Lid = torch.zeros_like(Gid).to(torch.int32).cuda() + 1
# print(Gid)
# print(Lid)
# Gid,Lid,uniTable = dgl.mapByNodeSet(nodeTable,uniTable,Gid,Lid,rhsNeed=False,include_rhs_in_lhs=False)
# print(nodeTable)
# print(uniTable)
# print(Gid)
# print(Lid)


"""
dgl.findSameNode(tensor1,tensor2,indexTable1,indexTable2)
    在两个单调增的向量中寻找相同的值,并返回相同值的索引位置

    args:
        tensor1(int32,cuda)         : 用于比较的第一个向量
        tensor2(int32,cuda)         : 用于比较的第二个向量
        indexTable1(int32,cuda)     : 用于存储第一个向量的相同值索引
        indexTable2(int32,cuda)     : 用于存储第二个向量的相同值索引
    
    return:
        indexTable1(int32,cuda)         : 存储第一个向量的相同值索引
        indexTable2(int32,cuda)         : 存储第二个向量的相同值索引


"""
#
#
#
#
#
#
#
#



"""
dgl.per_pagerank(src,dst,degreeTable,nodeValue,nodeInfo)
    用于计算每个点的pagerank,并传递分区信息
    src -> dst
    
    args:
        src(int32,cuda)         : 表示图的起始列
        dst(int32,cuda)         : 表示图的终止列
        degreeTable(int32,cuda) : 表示图度信息
        nodeValue(int32,cuda)   : 表示图权值
        nodeInfo(int32,cuda)    : 表示图需要传递的信息
    
    return:
        None: 直接在nodeValue,nodeInfo中进行修改
"""
degreeTable = torch.Tensor([1,1,1,1,1,1,1,1,1,1,1,1]).to(torch.int32).cuda()
src = torch.Tensor([0,4,4,1,5,5,3,6,6,7,7,2]).to(torch.int32).cuda()
dst = torch.Tensor([4,8,5,5,8,6,6,7,8,8,4,7]).to(torch.int32).cuda()
nodeValue = torch.Tensor([0,0,1000000,0,0,0,0,0]).to(torch.int32).cuda()
nodeInfo = torch.Tensor([1,2,4,8,0,0,0,0,0,0,0,0]).to(torch.int32).cuda()
print("nodeInfo:",nodeInfo)
nodeValue,nodeInfo = dgl.per_pagerank(src,dst,degreeTable,nodeValue,nodeInfo)
print("nodeValue:",nodeValue)
print("nodeInfo:",nodeInfo)

"""
分区中
首先抽取对应id
随后抽取重要度
然后按照重要度进行重排
对重排结果分成不同的格内部
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
dgl.loss_csr(raw_ptr,new_ptr,raw_indice,new_indice)
    用于删除不必要的indice部分,new_ptr需要已经给出
    
    args:
        raw_ptr(int32,cuda)         : 
        new_ptr(int32,cuda)         : 
        raw_indice(int32,cuda)      : 
        new_indice(int32,cuda)      : 

    
    return:
        None: 直接在nodeValue,nodeInfo中进行修改
"""
# raw_ptr = torch.tensor([0, 2, 5, 8, 11, 14, 17, 20], dtype=torch.int32)
# raw_indice = torch.tensor([0, 1, 3, 2, 4, 5, 0, 1, 6, 2, 3, 4, 6, 0, 1, 5, 2, 3, 4], dtype=torch.int32)
# ptr_diff = torch.diff(raw_ptr)
# select_idx = torch.tensor([0, 2, 4], dtype=torch.int64)
# ptr_diff[select_idx] = 0 
# new_ptr = torch.cat((torch.zeros(1).to(torch.int32),torch.cumsum(ptr_diff,dim = 0).to(torch.int32)))
# new_indice = torch.zeros(new_ptr[-1].item()-1).to(torch.int32)
# raw_ptr = raw_ptr.cuda()
# raw_indice = raw_indice.cuda()
# new_ptr = new_ptr.cuda()
# new_indice = new_indice.cuda()
# print("raw_ptr: ",raw_ptr)
# print("raw_indice: ",raw_indice)
# dgl.loss_csr(raw_ptr,new_ptr,raw_indice,new_indice)
# print("new_ptr: ",new_ptr)
# print("new_indice: ",new_indice)


"""
dgl.cooTocsr(inptr,indice,addr,srcList,dstList)
    将coo转换为csr的格式
    压缩 src 保留 dst
"""

# raw_ptr = torch.tensor([0, 1, 3, 2, 4, 5, 0, 1, 6, 2, 3, 4, 6, 0, 1, 5, 2, 3, 4], dtype=torch.int32)
# raw_indice = torch.tensor([0, 1, 3, 2, 4, 5, 0, 1, 6, 2, 3, 4, 6, 0, 1, 5, 2, 3, 4], dtype=torch.int32)


"""
dgl.lpGraph(src,dst,nodeTable): 双向进行
    进行标签传递
"""
src = torch.Tensor([0,2,4,5,3,4,2,5]).to(torch.int32).cuda()
dst = torch.Tensor([1,3,7,6,4,2,1,3]).to(torch.int32).cuda()
nodeLabel = torch.Tensor([-1,-1,2,3,4,-1,-1,-1]).to(torch.int32).cuda()
print("nodeLabel :",nodeLabel)
dgl.lpGraph(src,dst,nodeLabel)
print("nodeLabel :",nodeLabel)


"""
dgl.bincount(nodelist,nodeTable)
    对离散值分布就行求解
"""
# src = torch.Tensor([0,2,4,5,3,4,2,5]).to(torch.int32).cuda()
# nodeTable = torch.zeros(torch.max(src).item()+1,dtype=torch.int32,device="cuda")
# print("nodeLabel :",nodeTable)
# dgl.bincount(src,nodeTable)
# print("nodeLabel :",nodeTable)