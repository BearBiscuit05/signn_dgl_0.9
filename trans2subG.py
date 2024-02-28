import numpy as np
import dgl
import torch
import os
import copy
import gc
import time
from scipy.sparse import csr_matrix,coo_matrix
import json
from tools import *


# =============== 1.partition

def acc_ana(tensor):
    num_ones = torch.sum(tensor == 1).item()  
    total_elements = tensor.numel()  
    percentage_ones = (num_ones / total_elements) * 100 
    print(f"only use by one train node : {percentage_ones:.2f}%")
    num_greater_than_1 = torch.sum(tensor > 1).item() 
    percentage_greater_than_1 = (num_greater_than_1 / total_elements) * 100
    print(f"use by multi train nodes : {percentage_greater_than_1:.2f}%")
    # edgeNUM = edgeTable.cpu().sum() - edgeNUM
    # print(f"edge add to subG : {edgeNUM} , {edgeNUM * 1.0 / allEdgeNUM * 100 :.2f}% of total edges")
    # print(f"after {index} BFS has {torch.nonzero(nodeTable).size(0)} nodes, "
    # f"{torch.nonzero(nodeTable).size(0) * 1.0 / maxID * 100 :.2f}% of total nodes")

RUNTIME = 0
SAVETIME = 0
MERGETIME = 0
MAXEDGE = 100000000
## bfs 遍历获取基础子图
def analysisG(graph,maxID,trainId=None,savePath=None):
    global RUNTIME
    global SAVETIME
    dst = torch.tensor(graph[::2])
    src = torch.tensor(graph[1::2])
    if trainId == None:
        trainId = torch.arange(int(maxID*0.01),dtype=torch.int64)
    nodeTable = torch.zeros(maxID,dtype=torch.int32)
    nodeTable[trainId] = 1

    batch_size = len(src) // MAXEDGE + 1
    src_batches = torch.chunk(src, batch_size, dim=0)
    dst_batches = torch.chunk(dst, batch_size, dim=0)
    batch = [src_batches, dst_batches]

    repeats = 3
    start = time.time()
    edgeTable = torch.zeros_like(src,dtype=torch.int32).cuda()
    edgeNUM = 0
    allEdgeNUM = src.numel()
    for index in range(1,repeats+1):
        acc_tabel = torch.zeros_like(nodeTable,dtype=torch.int32)
        offset = 0
        for src_batch,dst_batch in zip(*batch):
            tmp_nodeTabel = copy.deepcopy(nodeTable)
            tmp_nodeTabel = tmp_nodeTabel.cuda()
            src_batch = src_batch.cuda()
            dst_batch = dst_batch.cuda()
            dgl.fastFindNeigEdge(tmp_nodeTabel,edgeTable,src_batch, dst_batch, offset)
            offset += len(src_batch)
            tmp_nodeTabel = tmp_nodeTabel.cpu()
            acc_tabel = acc_tabel | tmp_nodeTabel
        nodeTable = acc_tabel
    edgeTable = edgeTable.cpu()
    graph = graph.reshape(-1,2)
    nodeSet =  torch.nonzero(nodeTable).reshape(-1).to(torch.int32)
    edgeTable = torch.nonzero(edgeTable).reshape(-1).to(torch.int32)
    selfLoop = np.repeat(nodeSet.to(torch.int32), 2)
    subGEdge = graph[edgeTable]
    RUNTIME += time.time()-start

    saveTime = time.time()
    checkFilePath(savePath)
    DataPath = savePath + f"/raw_G.bin"
    TrainPath = savePath + f"/raw_trainIds.bin"
    NodePath = savePath + f"/raw_nodes.bin"
    saveBin(nodeSet,NodePath)
    saveBin(selfLoop,DataPath)
    saveBin(subGEdge,DataPath,addSave=True)
    saveBin(trainId,TrainPath)
    SAVETIME += time.time()-saveTime
    return RUNTIME,SAVETIME

def PRgenG(RAWPATH,nodeNUM,partNUM,savePath=None):
    GRAPHPATH = RAWPATH + "/graph.bin"
    TRAINPATH = RAWPATH + "/trainIds.bin"

    for i in range(partNUM):
        PATH = savePath + f"/part{i}" 
        checkFilePath(PATH)
    
    graph = torch.tensor(np.fromfile(GRAPHPATH,dtype=np.int32))
    src,dst = graph[::2],graph[1::2]
    trainIds = torch.tensor(np.fromfile(TRAINPATH,dtype=np.int64))
    edgeTable = torch.zeros_like(src).to(torch.int32)
    template_array = torch.zeros(nodeNUM,dtype=torch.int32)

    inNodeTable = copy.deepcopy(template_array)
    outNodeTable = copy.deepcopy(template_array)
    inNodeTable,outNodeTable = dgl.sumDegree(inNodeTable.cuda(),outNodeTable.cuda(),src.cuda(),dst.cuda())
    inNodeTable = inNodeTable.cpu()
    outNodeTable = outNodeTable.cpu()

    nodeValue = copy.deepcopy(template_array)
    nodeInfo = copy.deepcopy(template_array)
    nodeValue[trainIds] = 10000

    # random method
    shuffled_indices = torch.randperm(trainIds.size(0))
    r_trainId = trainIds[shuffled_indices]
    trainBatch = torch.chunk(r_trainId, partNUM, dim=0)

    for index,ids in enumerate(trainBatch):
        info = 1 << index
        nodeInfo[ids] = info
        # 存储训练集
        PATH = savePath + f"/part{index}" 
        TrainPath = PATH + f"/raw_trainIds.bin"
        saveBin(ids,TrainPath)
    
    dst = dst.cuda()
    src = src.cuda()
    edgeTable, inNodeTable = edgeTable.cuda(), inNodeTable.cuda()
    nodeValue, nodeInfo = nodeValue.cuda(), nodeInfo.cuda()
    for _ in range(3):    
        dgl.per_pagerank(dst,src,edgeTable,inNodeTable,nodeValue,nodeInfo)
    dst,src = dst.cpu(),src.cpu()
    edgeTable,inNodeTable = edgeTable.cpu(),inNodeTable.cpu()
    nodeValue,nodeInfo = nodeValue.cpu(),nodeInfo.cpu()

    for bit_position in range(partNUM):
        nodeIndex = (nodeInfo & (1 << bit_position)) != 0
        edgeIndex = (edgeTable & (1 << bit_position)) != 0
        nid = torch.nonzero(nodeIndex).reshape(-1).to(torch.int32)
        eid = torch.nonzero(edgeIndex).reshape(-1)
        graph = graph.reshape(-1,2)
        subEdge = graph[eid]
        partValue = nodeValue[nid]    
        selfLoop = np.repeat(nid.to(torch.int32), 2)
        
        PATH = savePath + f"/part{bit_position}" 
        TrainPath = PATH + f"/raw_trainIds.bin"
        DataPath = PATH + f"/raw_G.bin"
        NodePath = PATH + f"/raw_nodes.bin"
        PRvaluePath = PATH + f"/raw_value.bin"
        saveBin(nid,NodePath)
        saveBin(selfLoop,DataPath)
        saveBin(subEdge,DataPath,addSave=True)
        saveBin(partValue,PRvaluePath)

# =============== 2.graphToSub    
def nodeShuffle(raw_node,raw_graph,savePath=None,saveRes=False):
    torch.cuda.empty_cache()
    gc.collect()
    srcs = raw_graph[1::2]
    dsts = raw_graph[::2]
    if isinstance(data, np.ndarray):
        raw_node = torch.tensor(raw_node).to(torch.int32).cuda()
        srcs_tensor = torch.from_numpy(srcs).to(torch.int32)
        dsts_tensor = torch.from_numpy(dsts).to(torch.int32)
    else:
        raw_node = raw_node.to(torch.int32).cuda()
        srcs_tensor = srcs.to(torch.int32)
        dsts_tensor = dsts.to(torch.int32)
    uni = torch.ones(len(raw_node)*2).to(torch.int32).cuda()
    print("begin shuffle...")
    
    batch_size = len(srcs) // MAXEDGE + 1
    src_batches = list(torch.chunk(srcs_tensor, batch_size, dim=0))
    dst_batches = list(torch.chunk(dsts_tensor, batch_size, dim=0))

    batch = [src_batches, dst_batches]
    for index,(src_batch,dst_batch) in enumerate(zip(*batch)):
        src_batch = src_batch.cuda()
        dst_batch = dst_batch.cuda()
        srcShuffled,dstShuffled,uni = dgl.mapByNodeSet(raw_node,uni,src_batch,dst_batch)
        src_batches[index] = srcShuffled.cpu()
        dst_batches[index] = dstShuffled.cpu()
    srcs_tensor = torch.cat(src_batches)
    dsts_tensor = torch.cat(dst_batches)
    # print(srcs_tensor)
    # print(dsts_tensor)
    # exit(-1)
    uni = uni.cpu()
    if saveRes:
        graph = torch.stack((srcShuffled,dstShuffled),dim=1)
        graph = graph.reshape(-1).numpy()
        graph.tofile(savePath)
    print("shuffle end...")
    return srcs_tensor,dsts_tensor,uni

def trainIdxSubG(subGNode,trainSet):
    trainSet = torch.tensor(trainSet).to(torch.int32)
    Lid = torch.zeros_like(trainSet).to(torch.int32).cuda()
    dgl.mapLocalId(subGNode.cuda(),trainSet.cuda(),Lid)
    Lid = Lid.cpu().to(torch.int64)
    return Lid

def coo2csr(srcs,dsts):
    g = dgl.graph((srcs, dsts)).formats('csr')
    indptr, indices, _ = g.adj_sparse(fmt='csr')
    return indptr,indices
    # row,col = srcs,dsts
    # data = np.ones(len(col),dtype=np.int32)
    # m = csr_matrix((data, (row.numpy(), col.numpy())))
    # return m.indptr,m.indices

def rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,LABELPATH,SAVEPATH,featLen):
    labels = np.fromfile(LABELPATH,dtype=np.int64)
    for i in range(partitionNUM):
        startTime = time.time()
        PATH = RAWDATAPATH + f"/part{i}" 
        rawDataPath = PATH + f"/raw_G.bin"
        rawTrainPath = PATH + f"/raw_trainIds.bin"
        rawNodePath = PATH + f"/raw_nodes.bin"
        SubTrainIdPath = PATH + "/trainIds.bin"
        SubIndptrPath = PATH + "/indptr.bin"
        SubIndicesPath = PATH + "/indices.bin"
        SubLabelPath = PATH + "/labels.bin"
        checkFilePath(PATH)
        data = np.fromfile(rawDataPath,dtype=np.int32)
        node = np.fromfile(rawNodePath,dtype=np.int32)
        trainidx = np.fromfile(rawTrainPath,dtype=np.int64)
        srcShuffled,dstShuffled,uni = nodeShuffle(node,data)
        subLabel = labels[uni.to(torch.int64)]
        indptr, indices = coo2csr(srcShuffled,dstShuffled)
        trainidx = trainIdxSubG(uni,trainidx)
        saveBin(subLabel,SubLabelPath)
        saveBin(trainidx,SubTrainIdPath)
        saveBin(indptr,SubIndptrPath)
        saveBin(indices,SubIndicesPath)

def raw2subGwithPR(RAWDATAPATH,partitionNUM,FEATPATH,LABELPATH,featLen):
    labels = np.fromfile(LABELPATH,dtype=np.int64)
    feat = np.fromfile(FEATPATH,dtype=np.float32).reshape(-1,featLen)
    for i in range(partitionNUM):
        PATH = RAWDATAPATH + f"/part{i}" 
        # raw
        rawDataPath = PATH + f"/raw_G.bin"
        rawTrainPath = PATH + f"/raw_trainIds.bin"
        rawNodePath = PATH + f"/raw_nodes.bin"
        rawPRvaluePath = PATH + f"/raw_value.bin"
        # new
        SubTrainIdPath = PATH + "/trainIds.bin"
        SubIndptrPath = PATH + "/indptr.bin"
        SubIndicesPath = PATH + "/indices.bin"
        SubLabelPath = PATH + "/labels.bin"
        SubGRandomPath = PATH + "/randomG.bin"

        graph = torch.tensor(np.fromfile(rawDataPath,dtype=np.int32))
        nodeIds = torch.tensor(np.fromfile(rawNodePath,dtype=np.int64))
        trainidx = torch.tensor(np.fromfile(rawTrainPath,dtype=np.int64))
        prValue = torch.tensor(np.fromfile(rawPRvaluePath,dtype=np.int32))
        subLabel = labels[nodeIds]
        subfeat = feat[nodeIds]

        r_prValue, indices = torch.sort(prValue, descending=True)
        nodeIds = nodeIds[indices]
        subfeat = subfeat[indices]
        subLabel = subLabel[indices]
        
        srcShuffled,dstShuffled,uni = nodeShuffle(nodeIds,graph)
        trainidx = trainIdxSubG(uni,trainidx)

        sort_dstList,indice = torch.sort(dstShuffled,dim=0) # 有待提高
        sort_srcList = srcShuffled[indice]
        
        nodeSize = uni.shape[0]
        fix_NUM = int(nodeSize * 0.1)
        position = torch.searchsorted(sort_dstList, fix_NUM)
        fix_indice = sort_srcList[:position]
        fix_inptr = torch.cat([torch.Tensor([0]).to(torch.int32),torch.cumsum(torch.bincount(sort_dstList[:position]), dim=0)]).to(torch.int32)
        
        random_dst = sort_dstList[position:]
        random_src = sort_srcList[position:]
        randomG = torch.stack((random_dst, random_src), dim=0).to(torch.int32)

        saveBin(subLabel,SubLabelPath)
        saveBin(trainidx,SubTrainIdPath)
        saveBin(fix_inptr,SubIndptrPath)
        saveBin(fix_indice,SubIndicesPath)
        saveBin(randomG,SubGRandomPath)

# =============== 3.featTrans
def featSlice(FEATPATH,beginIndex,endIndex,featLen):
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    subFeat = np.fromfile(FEATPATH, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset)
    return subFeat.reshape(-1,featLen)

def sliceIds(Ids,sliceTable):
    beginIndex = 0
    ans = []
    for tar in sliceTable[1:]:
        position = torch.searchsorted(Ids, tar)
        slice = Ids[beginIndex:position]
        ans.append(slice)
        beginIndex = position
    return ans

def genSubGFeat(SAVEPATH,FEATPATH,partNUM,nodeNUM,sliceNUM,featLen):
    # 获得切片
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM
    print("bound:",boundList)

    idsSliceList = [[] for i in range(partNUM)]
    for i in range(partNUM):
        file = SAVEPATH + f"/part{i}/raw_nodes.bin"
        ids = torch.tensor(np.fromfile(file,dtype=np.int32))
        idsSliceList[i] = sliceIds(ids,boundList)
    
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen)
        for index in range(partNUM):
            fileName = SAVEPATH + f"/part{index}/feat.bin"
            SubIdsList = idsSliceList[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            subFeat = sliceFeat[t_SubIdsList]
            saveBin(subFeat,fileName,addSave=sliceIndex)


# =============== 4. randomGen
def randomGen(PATH,partid,nodeNUM):
    raw_ptr = torch.tensor(np.fromfile(PATH + f"/part{partid}/indptr.bin",dtype=np.int32))
    raw_indice = torch.tensor(np.fromfile(PATH + f"/part{partid}/indices.bin",dtype=np.int32))
    randomG = torch.tensor(np.fromfile(PATH + f"/part{partid}/randomG.bin",dtype=np.int32))
    
    mapTable = torch.zeros(nodeNUM).to(torch.int32) - 1
    fix_NUM = raw_ptr.shape[0] - 1
    fix_index = torch.arange(fix_NUM).to(torch.int32)
    mapTable[fix_index.to(torch.int64)] = fix_index

    coo_src= randomG[::2][:4000000]
    coo_dst = randomG[1::2][:4000000]

    bincount = torch.bincount(coo_dst)
    gid = torch.nonzero(bincount).reshape(-1)
    lid = torch.arange(len(gid)).to(torch.int32) + fix_NUM
    mapTable[gid] = lid
    cumList = bincount[gid]

    new_inptr = torch.cumsum(cumList, dim=0).to(torch.int32) + raw_ptr[-1]

    inptr = torch.cat([raw_ptr,new_inptr])
    indice = torch.cat([raw_indice,coo_src])

    return inptr,indice,mapTable


if __name__ == '__main__':
    JSONPATH = "./datasetInfo.json"
    partitionNUM = 4
    sliceNUM = 5
    with open(JSONPATH, 'r') as file:
        data = json.load(file)
    datasetName = ["PD"] 

    for NAME in datasetName:
        subGSavePath = data[NAME]["processedPath"]
        GRAPHPATH = data[NAME]["rawFilePath"]
        maxID = data[NAME]["nodes"]

        PRgenG(GRAPHPATH,maxID,partitionNUM,savePath=subGSavePath)

        print("PR run success...")
        FEATPATH = data[NAME]["rawFilePath"] + "/feat.bin"
        LABELPATH = data[NAME]["rawFilePath"] + "/labels.bin"
        featLen = data[NAME]["featLen"]

        raw2subGwithPR(subGSavePath,partitionNUM,FEATPATH,LABELPATH,featLen)

        randomGen(subGSavePath,0,maxID)
        

    # for NAME in datasetName:
    #     GRAPHPATH = data[NAME]["rawFilePath"]
    #     maxID = data[NAME]["nodes"]
    #     subGSavePath = data[NAME]["processedPath"]
    #     trainId = torch.tensor(np.fromfile(GRAPHPATH + "/trainIds.bin",dtype=np.int64))
    #     shuffled_indices = torch.randperm(trainId.size(0))
    #     trainId = trainId[shuffled_indices]
    #     trainBatch = torch.chunk(trainId, partitionNUM, dim=0)

    #     graph = np.fromfile(GRAPHPATH+"/graph.bin",dtype=np.int32)
    #     startTime = time.time()
    #     for index,trainids in enumerate(trainBatch):
    #         analysisG(graph,maxID,trainId=trainids,savePath=subGSavePath+f"/part{index}")
    #     print(f"run time cost:{RUNTIME:.3f}")
    #     print(f"save time cost:{SAVETIME:.3f}")
    #     print(f"partition all cost:{time.time()-startTime:.3f}")
    
    # for NAME in datasetName:
    #     RAWDATAPATH = data[NAME]["processedPath"]
    #     FEATPATH = data[NAME]["rawFilePath"] + "/feat.bin"
    #     LABELPATH = data[NAME]["rawFilePath"] + "/labels.bin"
    #     SAVEPATH = data[NAME]["processedPath"]
    #     nodeNUM = data[NAME]["nodes"]
    #     featLen = data[NAME]["featLen"]
        
    #     MERGETIME = time.time()
    #     rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,LABELPATH,SAVEPATH,featLen)
    #     print(f"trans graph cost time{time.time() - MERGETIME:.3f}...")
    #     FEATTIME = time.time()
    #     genSubGFeat(SAVEPATH,FEATPATH,partitionNUM,nodeNUM,sliceNUM,featLen)
    #     print(f"graph feat gen cost time{time.time() - FEATTIME:.3f}...")