/*!
 *  Copyright 2019-2021 Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * \file graph/transform/to_bipartite.cc
 * \brief Convert a graph to a bipartite-structured graph.
 */

#include "to_bipartite.h"

#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/container.h>
#include <vector>
#include <tuple>
#include <utility>
#include "../../array/cpu/array_utils.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

namespace {

// Since partial specialization is not allowed for functions, use this as an
// intermediate for ToBlock where XPU = kDLCPU.
template<typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlockCPU(HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* const lhs_nodes_ptr) {
  std::vector<IdArray>& lhs_nodes = *lhs_nodes_ptr;
  const bool generate_lhs_nodes = lhs_nodes.empty();

  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();
  std::vector<EdgeArray> edge_arrays(num_etypes);

  CHECK(rhs_nodes.size() == static_cast<size_t>(num_ntypes))
    << "rhs_nodes not given for every node type";

  const std::vector<IdHashMap<IdType>> rhs_node_mappings(rhs_nodes.begin(), rhs_nodes.end());
  std::vector<IdHashMap<IdType>> lhs_node_mappings;

  if (generate_lhs_nodes) {
  // build lhs_node_mappings -- if we don't have them already
    if (include_rhs_in_lhs)
      lhs_node_mappings = rhs_node_mappings;  // copy
    else
      lhs_node_mappings.resize(num_ntypes);
  } else {
    lhs_node_mappings = std::vector<IdHashMap<IdType>>(lhs_nodes.begin(), lhs_nodes.end());
  }


  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    if (!aten::IsNullArray(rhs_nodes[dsttype])) {
      const EdgeArray& edges = graph->Edges(etype);
      if (generate_lhs_nodes) {
        lhs_node_mappings[srctype].Update(edges.src);
      }
      edge_arrays[etype] = edges;
    }
  }

  std::vector<int64_t> num_nodes_per_type;
  num_nodes_per_type.reserve(2 * num_ntypes);

  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      num_ntypes * 2, etypes.src, new_dst);

  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype)
    num_nodes_per_type.push_back(lhs_node_mappings[ntype].Size());
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype)
    num_nodes_per_type.push_back(rhs_node_mappings[ntype].Size());

  std::vector<HeteroGraphPtr> rel_graphs;
  std::vector<IdArray> induced_edges;
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    const IdHashMap<IdType> &lhs_map = lhs_node_mappings[srctype];
    const IdHashMap<IdType> &rhs_map = rhs_node_mappings[dsttype];
    if (rhs_map.Size() == 0) {
      // No rhs nodes are given for this edge type. Create an empty graph.
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_map.Size(), rhs_map.Size(),
          aten::NullArray(), aten::NullArray()));
      induced_edges.push_back(aten::NullArray());
    } else {
      IdArray new_src = lhs_map.Map(edge_arrays[etype].src, -1);
      IdArray new_dst = rhs_map.Map(edge_arrays[etype].dst, -1);
      // Check whether there are unmapped IDs and raise error.
      for (int64_t i = 0; i < new_dst->shape[0]; ++i)
        CHECK_NE(new_dst.Ptr<IdType>()[i], -1)
          << "Node " << edge_arrays[etype].dst.Ptr<IdType>()[i] << " does not exist"
          << " in `rhs_nodes`. Argument `rhs_nodes` must contain all the edge"
          << " destination nodes.";
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_map.Size(), rhs_map.Size(),
          new_src, new_dst));
      induced_edges.push_back(edge_arrays[etype].id);
    }
  }

  const HeteroGraphPtr new_graph = CreateHeteroGraph(
      new_meta_graph, rel_graphs, num_nodes_per_type);

  if (generate_lhs_nodes) {
    CHECK_EQ(lhs_nodes.size(), 0) << "InteralError: lhs_nodes should be empty "
        "when generating it.";
    for (const IdHashMap<IdType> &lhs_map : lhs_node_mappings)
      lhs_nodes.push_back(lhs_map.Values());
  }
  return std::make_tuple(new_graph, induced_edges);
}

}  // namespace

template<>
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlock<kDLCPU, int32_t>(HeteroGraphPtr graph,
                         const std::vector<IdArray> &rhs_nodes,
                         bool include_rhs_in_lhs,
                         std::vector<IdArray>* const lhs_nodes) {
  return ToBlockCPU<int32_t>(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

template<>
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlock<kDLCPU, int64_t>(HeteroGraphPtr graph,
                         const std::vector<IdArray> &rhs_nodes,
                         bool include_rhs_in_lhs,
                         std::vector<IdArray>* const lhs_nodes) {
  return ToBlockCPU<int64_t>(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

#ifdef DGL_USE_CUDA

// Forward declaration of GPU ToBlock implementations - actual implementation is in
// ./cuda/cuda_to_block.cu
// This is to get around the broken name mangling in VS2019 CL 16.5.5 + CUDA 11.3
// which complains that the two template specializations have the same signature.
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlockGPU32(HeteroGraphPtr, const std::vector<IdArray>&, bool, std::vector<IdArray>* const);
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlockGPU64(HeteroGraphPtr, const std::vector<IdArray>&, bool, std::vector<IdArray>* const);
std::tuple<IdArray,IdArray,IdArray>
ReMapIds(IdArray &lhs_nodes,IdArray &rhs_nodes,IdArray &uni_nodes,bool include_rhs_in_lhs);
std::tuple<IdArray,IdArray,IdArray>
mapByNodeToEdge(IdArray &lhsNode,IdArray &rhsNode,IdArray &uniTable,IdArray &srcList,IdArray &dstList,bool include_rhs_in_lhs);
void
c_loadGraphHalo(IdArray &indptr,IdArray &indices,IdArray &edges,IdArray &bound,int gap);
void
c_FindNeighborByBfs(IdArray &nodeTable,IdArray &tmpTable,IdArray &srcList,IdArray &dstList,bool acc);
void
c_FindNeigEdgeByBfs(IdArray &nodeTable,IdArray &tmpNodeTable,IdArray &edgeTable,IdArray &srcList,IdArray &dstList,int64_t offset,int32_t loopFlag);
void
c_maplocalIds(IdArray &nodeTable,IdArray &Gids,IdArray &Lids);
void
c_findSameNode(IdArray &tensor1,IdArray &tensor2,IdArray &indexTable1,IdArray &indexTable2);
void
c_sumDegree(IdArray &nodeTabel,IdArray &srcList,IdArray &dstList);
void
c_calculateP(IdArray &DegreeTabel,IdArray &PTabel,IdArray &srcList,IdArray &dstList,int64_t fanout);

template<>
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlock<kDLGPU, int32_t>(HeteroGraphPtr graph,
                         const std::vector<IdArray> &rhs_nodes,
                         bool include_rhs_in_lhs,
                         std::vector<IdArray>* const lhs_nodes) {
  return ToBlockGPU32(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

template<>
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlock<kDLGPU, int64_t>(HeteroGraphPtr graph,
                         const std::vector<IdArray> &rhs_nodes,
                         bool include_rhs_in_lhs,
                         std::vector<IdArray>* const lhs_nodes) {
  return ToBlockGPU64(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

template<>
std::tuple<IdArray, IdArray, IdArray>
Trans2ReMap<kDLGPU, int32_t>(
                         IdArray &lhs_nodes,
                         IdArray &rhs_nodes,
                         IdArray &uni_nodes,
                         bool include_rhs_in_lhs
                         ) {
  return ReMapIds(lhs_nodes,rhs_nodes,uni_nodes,include_rhs_in_lhs);
}

template<>
std::tuple<IdArray, IdArray, IdArray>
mapByNodeTable<kDLGPU, int32_t>(
                         IdArray &lhsNode,
                         IdArray &rhsNode,
                         IdArray &uniTable,
                         IdArray &srcList,
                         IdArray &dstList,
                         bool include_rhs_in_lhs
                         ) {
  return mapByNodeToEdge(lhsNode,rhsNode,uniTable,srcList,dstList,include_rhs_in_lhs);
}

template<>
void
loadHalo2Graph<kDLGPU, int32_t>(
                         IdArray &indptr,
                         IdArray &indices,
                         IdArray &edges,
                         IdArray &bound,
                         int gap
                         ) {
  return c_loadGraphHalo(indptr,indices,edges,bound,gap);
}

template<>
void
FindNeighbor<kDLGPU, int32_t>(
  IdArray &nodeTable,
  IdArray &srcList,
  IdArray &dstList,
  bool acc) {
  IdArray tmpTable = Full(0,nodeTable->shape[0],srcList->ctx);
  c_FindNeighborByBfs(nodeTable,tmpTable,srcList,dstList,acc);
} 

template<>
void
FindNeigEdge<kDLGPU, int32_t>(
  IdArray &nodeTable,
  IdArray &edgeTable,
  IdArray &srcList,
  IdArray &dstList,
  int64_t offset,
  int32_t loopFlag) {
    IdArray tmpNodeTable = Full(0,nodeTable->shape[0],srcList->ctx);
    c_FindNeigEdgeByBfs(nodeTable,tmpNodeTable,edgeTable,srcList,dstList,offset,loopFlag);
  }

template<>
void
maplocalIds<kDLGPU, int32_t>(
  IdArray &nodeTable,
  IdArray &Gids,
  IdArray &Lids) {
    c_maplocalIds(nodeTable,Gids,Lids);
  }

template<>
void
findSameNode<kDLGPU, int32_t>(
  IdArray &tensor1,
  IdArray &tensor2,
  IdArray &indexTable1,
  IdArray &indexTable2) {
    c_findSameNode(tensor1,tensor2,indexTable1,indexTable2);
  }

template<>
void
sumDegree<kDLGPU, int32_t>(
  IdArray &nodeTabel,
  IdArray &srcList,
  IdArray &dstList) {
    c_sumDegree(nodeTabel,srcList,dstList);
  }

template<>
void
calculateP<kDLGPU, int32_t>(
  IdArray &DegreeTabel,
  IdArray &PTabel,
  IdArray &srcList,
  IdArray &dstList,
  int64_t fanout) {
    c_calculateP(DegreeTabel,PTabel,srcList,dstList,fanout);
  }

#endif  // DGL_USE_CUDA

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToBlock")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef graph_ref = args[0];
    const std::vector<IdArray> &rhs_nodes = ListValueToVector<IdArray>(args[1]);
    const bool include_rhs_in_lhs = args[2];
    std::vector<IdArray> lhs_nodes = ListValueToVector<IdArray>(args[3]);

    HeteroGraphPtr new_graph;
    std::vector<IdArray> induced_edges;

    ATEN_XPU_SWITCH_CUDA(graph_ref->Context().device_type, XPU, "ToBlock", {
      ATEN_ID_TYPE_SWITCH(graph_ref->DataType(), IdType, {
      std::tie(new_graph, induced_edges) = ToBlock<XPU, IdType>(
          graph_ref.sptr(), rhs_nodes, include_rhs_in_lhs,
          &lhs_nodes);
      });
    });

    List<Value> lhs_nodes_ref;
    for (IdArray &array : lhs_nodes)
      lhs_nodes_ref.push_back(Value(MakeValue(array)));
    List<Value> induced_edges_ref;
    for (IdArray &array : induced_edges)
      induced_edges_ref.push_back(Value(MakeValue(array)));

    List<ObjectRef> ret;
    ret.push_back(HeteroGraphRef(new_graph));
    ret.push_back(lhs_nodes_ref);
    ret.push_back(induced_edges_ref);

    *rv = ret;
  });

DGL_REGISTER_GLOBAL("transform._CAPI_ReMappingId")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray graph_src_nodes = args[0];
    IdArray graph_dst_nodes =args[1];
    IdArray unqiueSpace = args[2];
    IdArray new_src;
    IdArray new_dst;
    IdArray unqiue_nodes;
    bool include_rhs_in_lhs = true;
    std::tie(new_src, new_dst , unqiue_nodes) = Trans2ReMap<kDLGPU, int32_t>(
          graph_src_nodes,graph_dst_nodes,unqiueSpace,include_rhs_in_lhs); // 新版本存在更改
    *rv = ConvertNDArrayVectorToPackedFunc({new_src, new_dst,unqiue_nodes});
  });

DGL_REGISTER_GLOBAL("transform._CAPI_loadHalo")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray indptr = args[0];
    IdArray indices =args[1];
    IdArray edges = args[2];
    IdArray bound = args[3];
    int gap = args[4];
    
    loadHalo2Graph<kDLGPU, int32_t>(indptr,indices,edges,bound,gap); 
    *rv = ConvertNDArrayVectorToPackedFunc({indptr, indices});
  });

DGL_REGISTER_GLOBAL("transform._CAPI_fastFindNeighbor")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray nodeTable = args[0];
    IdArray srcList =args[1];
    IdArray dstList = args[2];
    bool acc = args[3];
    FindNeighbor<kDLGPU, int32_t>(nodeTable,srcList,dstList,acc); 
    *rv = ConvertNDArrayVectorToPackedFunc({nodeTable});
  });

DGL_REGISTER_GLOBAL("transform._CAPI_fastFindNeigEdge")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray nodeTable = args[0];
    IdArray edgeTable = args[1];
    IdArray srcList =args[2];
    IdArray dstList = args[3];
    int64_t offset = args[4];
    int32_t loopFlag = args[5];
    FindNeigEdge<kDLGPU, int32_t>(nodeTable,edgeTable,srcList,dstList,offset,loopFlag); 
    *rv = ConvertNDArrayVectorToPackedFunc({nodeTable,edgeTable});
  });

DGL_REGISTER_GLOBAL("transform._CAPI_MaplocalId")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray nodeTable = args[0];
    IdArray Gids =args[1];
    IdArray Lids = args[2];
    maplocalIds<kDLGPU, int32_t>(nodeTable,Gids,Lids); 
    *rv = ConvertNDArrayVectorToPackedFunc({Lids});
  });

DGL_REGISTER_GLOBAL("transform._CAPI_MapByNodeSet")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray lhsNodes = args[0];
    IdArray rhsNodes = args[1];
    IdArray uniTabel =args[2];
    IdArray srcList =args[3];
    IdArray dstList = args[4];
    IdArray new_src;
    IdArray new_dst;
    IdArray unqiue_nodes;
    bool include_rhs_in_lhs = true;
    std::tie(new_src, new_dst , unqiue_nodes) = 
      mapByNodeTable<kDLGPU, int32_t>(lhsNodes,rhsNodes,uniTabel,srcList,dstList,include_rhs_in_lhs); 
    *rv = ConvertNDArrayVectorToPackedFunc({new_src, new_dst , unqiue_nodes});
  });


DGL_REGISTER_GLOBAL("transform._CAPI_FindSameNode")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray tensor1 = args[0];
    IdArray tensor2 = args[1];
    IdArray indexTable1 =args[2];
    IdArray indexTable2 =args[3];
    findSameNode<kDLGPU, int32_t>(tensor1,tensor2,indexTable1,indexTable2);
    *rv = ConvertNDArrayVectorToPackedFunc({indexTable1, indexTable2});
  });

DGL_REGISTER_GLOBAL("transform._CAPI_SumDegree")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray nodeTabel = args[0];
    IdArray srcList = args[1];
    IdArray dstList =args[2];
    sumDegree<kDLGPU, int32_t>(nodeTabel,srcList,dstList);
    *rv = ConvertNDArrayVectorToPackedFunc({nodeTabel});
  });


DGL_REGISTER_GLOBAL("transform._CAPI_CalculateP")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray DegreeTabel = args[0];
    IdArray PTabel = args[1];
    IdArray srcList =args[2];
    IdArray dstList = args[3];
    int64_t fanout = args[4];
    calculateP<kDLGPU, int32_t>(DegreeTabel,PTabel,srcList,dstList,fanout);
    *rv = ConvertNDArrayVectorToPackedFunc({PTabel});
  });



};  // namespace transform

};  // namespace dgl
