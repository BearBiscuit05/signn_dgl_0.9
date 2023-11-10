/*!
 *  Copyright 2020-2021 Contributors
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
 * \file graph/transform/cuda/cuda_to_block.cu
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */


#include <dgl/runtime/device_api.h>
#include <dgl/immutable_graph.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <utility>
#include <algorithm>
#include <memory>
#include <chrono>
#include <cub/cub.cuh>
#include "../../../runtime/cuda/cuda_common.h"
#include "../../heterograph.h"
#include "../to_bipartite.h"
#include "cuda_map_edges.cuh"

using namespace dgl::aten;
using namespace dgl::runtime::cuda;
using namespace dgl::transform::cuda;

namespace dgl {
namespace transform {

namespace {

template<typename IdType>
class DeviceNodeMapMaker {
 public:
  DeviceNodeMapMaker(
      const std::vector<int64_t>& maxNodesPerType) :
      max_num_nodes_(0) {
    max_num_nodes_ = *std::max_element(maxNodesPerType.begin(),
        maxNodesPerType.end());
  }

  /**
  * \brief This function builds node maps for each node type, preserving the
  * order of the input nodes. Here it is assumed the lhs_nodes are not unique,
  * and thus a unique list is generated.
  *
  * \param lhs_nodes The set of source input nodes.
  * \param rhs_nodes The set of destination input nodes.
  * \param node_maps The node maps to be constructed.
  * \param count_lhs_device The number of unique source nodes (on the GPU).
  * \param lhs_device The unique source nodes (on the GPU).
  * \param stream The stream to operate on.
  */
  void Make(
      const std::vector<IdArray>& lhs_nodes,
      const std::vector<IdArray>& rhs_nodes,
      DeviceNodeMap<IdType> * const node_maps,
      int64_t * const count_lhs_device,
      std::vector<IdArray>* const lhs_device,
      cudaStream_t stream) {
    const int64_t num_ntypes = lhs_nodes.size() + rhs_nodes.size();

    CUDA_CALL(cudaMemsetAsync(
      count_lhs_device,
      0,
      num_ntypes*sizeof(*count_lhs_device),
      stream));

    // possibly dublicate lhs nodes
    const int64_t lhs_num_ntypes = static_cast<int64_t>(lhs_nodes.size());
    for (int64_t ntype = 0; ntype < lhs_num_ntypes; ++ntype) {
      const IdArray& nodes = lhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDLGPU);
        node_maps->LhsHashTable(ntype).FillWithDuplicates(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            (*lhs_device)[ntype].Ptr<IdType>(),
            count_lhs_device+ntype,
            stream);
      }
    }

    // unique rhs nodes
    const int64_t rhs_num_ntypes = static_cast<int64_t>(rhs_nodes.size());
    for (int64_t ntype = 0; ntype < rhs_num_ntypes; ++ntype) {
      const IdArray& nodes = rhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        node_maps->RhsHashTable(ntype).FillWithUnique(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            stream);
      }
    }
  }

  /**
  * \brief This function builds node maps for each node type, preserving the
  * order of the input nodes. Here it is assumed both lhs_nodes and rhs_nodes
  * are unique.
  *
  * \param lhs_nodes The set of source input nodes.
  * \param rhs_nodes The set of destination input nodes.
  * \param node_maps The node maps to be constructed.
  * \param stream The stream to operate on.
  */
  void Make(
      const std::vector<IdArray>& lhs_nodes,
      const std::vector<IdArray>& rhs_nodes,
      DeviceNodeMap<IdType> * const node_maps,
      cudaStream_t stream) {
    const int64_t num_ntypes = lhs_nodes.size() + rhs_nodes.size();

    // unique lhs nodes
    const int64_t lhs_num_ntypes = static_cast<int64_t>(lhs_nodes.size());
    for (int64_t ntype = 0; ntype < lhs_num_ntypes; ++ntype) {
      const IdArray& nodes = lhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDLGPU);
        node_maps->LhsHashTable(ntype).FillWithUnique(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            stream);
      }
    }

    // unique rhs nodes
    const int64_t rhs_num_ntypes = static_cast<int64_t>(rhs_nodes.size());
    for (int64_t ntype = 0; ntype < rhs_num_ntypes; ++ntype) {
      const IdArray& nodes = rhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        node_maps->RhsHashTable(ntype).FillWithUnique(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            stream);
      }
    }
  }

  void Make(
      IdArray& lhs_nodes,
      IdArray& rhs_nodes,
      DeviceNodeMap<IdType> * const node_maps,
      int64_t * const count_lhs_device,
      IdArray& lhs_device,
      cudaStream_t stream) {
    const int64_t num_ntypes = 2;

    CUDA_CALL(cudaMemsetAsync(
      count_lhs_device,
      0,
      num_ntypes*sizeof(*count_lhs_device),
      stream));

    // possibly dublicate lhs nodes
    const int64_t lhs_num_ntypes = static_cast<int64_t>(1);
    const int64_t rhs_num_ntypes = static_cast<int64_t>(1);
    
    for (int64_t ntype = 0; ntype < rhs_num_ntypes; ++ntype) {
      const IdArray& nodes = rhs_nodes;
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDLGPU);
        node_maps->RhsHashTable(ntype).FillWithDuplicates(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            lhs_device.Ptr<IdType>(),
            count_lhs_device+ntype,
            stream);
      }
    }
    
    for (int64_t ntype = 0; ntype < lhs_num_ntypes; ++ntype) {
      const IdArray& nodes = lhs_nodes;
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDLGPU);
        node_maps->LhsHashTable(ntype).FillWithDuplicates(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            lhs_device.Ptr<IdType>(),
            count_lhs_device+ntype,
            stream);
      }
    }
    
    
    
  }

 private:
  IdType max_num_nodes_;
};



template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void graph_halo_merge_kernel(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM,
    int gap,unsigned long random_states
) {
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    curandStateXORWOW_t local_state;
    curand_init(random_states+idx,0,0,&local_state);
    for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
        if (index < nodeNUM) {
            int rid = index;
            int startptr = bound[index*2+1];
            int endptr = bound[index*2+2];
            int space = endptr - startptr;
            int off = halo_bound[rid] + gap;
            int len = halo_bound[rid+1] - off;
            if (len > 0) {
                if (space < len) {
                    for (int j = 0; j < space; j++) {
                        int selected_j = curand(&local_state) % (len - j);
                        int selected_id = halos[off + selected_j];
                        edge[startptr++] = selected_id;
                        halos[off + selected_j] = halos[off+len-j-1];
                        halos[off+len-j-1] = selected_id;
                    }
                    bound[index*2+1] = startptr;
                } else {
                    for (int j = 0; j < len; j++) {
                        edge[startptr++] = halos[off + j];
                    }
                    bound[index*2+1] = startptr;
                }
            }
        }
    }
}

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void ToUseBfsKernel(
  int* nodeTable,
  int* tmpTable,
  int* srcList,
  int* dstList,
  int64_t edgeNUM,
  int64_t flag,
  bool acc) {
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += BLOCK_SIZE) {
      if (index < edgeNUM) {
        int srcID = srcList[index];
        int dstID = dstList[index];
        if(nodeTable[srcID] > 0 && nodeTable[dstID] == 0) {
          // src --> dst
          if (acc) {
            atomicAdd(&tmpTable[dstID], 1);
          } else {
            atomicExch(&tmpTable[dstID], flag);
          }
        }
      }
    }
}

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void ToMergeTable(
  int* nodeTable,
  int* tmpTable,
  int64_t nodeNUM,
  bool acc
) {
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  for (size_t index = threadIdx.x + block_start; index < block_end;
      index += BLOCK_SIZE) {
    if (index < nodeNUM) {
      if(tmpTable[index] > 0) {
        if (acc) {
          nodeTable[index] += tmpTable[index];
        } else {
          nodeTable[index] = tmpTable[index];
        }
      }
    }
  }
}

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void ToUseBfsWithEdgeKernel(
  int* nodeTable,
  int* tmpTable,
  int* edgeTable,
  int* srcList,
  int* dstList,
  int64_t edgeNUM,
  int64_t offset,
  int32_t loopFlag) {
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += BLOCK_SIZE) {
      if (index < edgeNUM) {
        int srcID = srcList[index];
        int dstID = dstList[index];
        // src --> dst
        if(nodeTable[srcID] > 0 && nodeTable[dstID] == 0) {
          atomicExch(&tmpTable[dstID], 1);
          edgeTable[offset + index] = loopFlag;
        } else if((nodeTable[srcID] > 0 && nodeTable[dstID] > 0) && (edgeTable[offset + index] == 0)) {
          edgeTable[offset + index] = 1;
        }
      }
    }
}


template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void ToMapLocalIdKernel(
  int* nodeTable,
  int* Gids,
  int* Lids,
  int64_t TableNUM,
  int64_t idsNUM
) {
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += BLOCK_SIZE) {
      if (index < idsNUM) {
        int Gid = Gids[index];
        for (int64_t i = 0 ; i < TableNUM ; i++) {
          if(Gid == nodeTable[i]) {
            Lids[index] = i;
            break;
          }
        }
      }
    }
}


template<int BLOCK_SIZE, int TILE_SIZE>
__device__ void findIndex(
    int* tensor,
    int* table,
    int* indexTable,
    int64_t tensorLen,
    int64_t tableLen
    ) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  for (size_t index = threadIdx.x + block_start; index < block_end;
      index += BLOCK_SIZE) {
    
    if (index < tensorLen) {  
      int id = tensor[index];
      int left = 0;
      int right=tableLen-1;
      while(left<=right)
      {
          int mid =(left+right)/2;
          if(id == table[mid]) {
            indexTable[index] = 1;
            // printf("number set !!! \n");
            break;
          }
          if(id>table[mid])
          {
              left = mid+1;
          }
          else
          {
              right = mid-1;
          }
      }
    }

  }
}

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void ToFindSameNodeKernel(
  int* in_t1,
  int* in_t2,
  int* in_table1,
  int* in_table2,
  int64_t t1NUM,
  int64_t t2NUM
) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);
  if (blockIdx.y == 0) {
    findIndex<BLOCK_SIZE, TILE_SIZE>(
        in_t1,  // find
        in_t2,  // table
        in_table1,
        t1NUM,
        t2NUM);
  } else {
    findIndex<BLOCK_SIZE, TILE_SIZE>(
        in_t2,  // find
        in_t1,  // table
        in_table2,
        t2NUM,
        t1NUM);
  }
}


template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void SumDegreeKernel(
  int* in_nodeTabel,
  int* out_nodeTabel,
  int* in_srcList,
  int* in_dstList,
  int64_t edgeNUM) {
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += BLOCK_SIZE) {
      if (index < edgeNUM) {
        int srcid = in_srcList[index];
        int dstid = in_dstList[index];
        atomicAdd(&in_nodeTabel[dstid], 1);
        atomicAdd(&out_nodeTabel[srcid], 1);
      }
    }
}

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void calculatePKernel(
  int* in_DegreeTabel,
  int* in_PTabel,
  int* in_srcList,
  int* in_dstList,
  int64_t edgeNUM,
  int64_t fanout){
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += BLOCK_SIZE) {
      if (index < edgeNUM) {
        int srcid = in_srcList[index];
        int dstid = in_dstList[index];
        if (srcid == dstid)
          continue;
        int degree = in_DegreeTabel[dstid];
        float srcP = in_PTabel[srcid] / 1000.0f;
        float dstP = in_PTabel[dstid] / 1000.0f;
        float notChoiceP = 1.0f - fmin(1.0f, fanout / (float)degree);
        srcP = (dstP + (1.0f - dstP) * notChoiceP);
        int r_srcP = int(srcP*1000);
        atomicMin(&in_PTabel[srcid], r_srcP);
      }
    }
  }

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void PPRkernel(
  int* src,
  int* dst,
  int* edgeTable,
  int* degreeTable,
  int* in_nodeValue,
  int* in_nodeInfo,
  int* in_tmpNodeValue,
  int* in_tmpNodeInfo,
  int64_t edgeNUM) {
    // src -> dst value and info
    float d = 0.85;
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += BLOCK_SIZE) {
      if (index < edgeNUM) {
        int srcId = src[index];
        int dstId = dst[index];
        int degree = degreeTable[srcId];
        float value = in_nodeValue[srcId];
        int src_info = in_nodeInfo[srcId];
        int dst_info = in_nodeInfo[dstId] | src_info;
        edgeTable[index] = src_info;
        if(value == 0.0f)
          continue;
        float con = value * d / (1000.0f * degree);
        atomicAdd(&in_tmpNodeValue[dstId], int(con*1000));
        atomicExch(&in_tmpNodeInfo[dstId],dst_info);
      }
    }
  }


// Since partial specialization is not allowed for functions, use this as an
// intermediate for ToBlock where XPU = kDLGPU.
template<typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlockGPU(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    const bool include_rhs_in_lhs,
    std::vector<IdArray>* const lhs_nodes_ptr) {
  std::vector<IdArray>& lhs_nodes = *lhs_nodes_ptr;
  const bool generate_lhs_nodes = lhs_nodes.empty();


  const auto& ctx = graph->Context();
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  CHECK_EQ(ctx.device_type, kDLGPU);
  for (const auto& nodes : rhs_nodes) {
    CHECK_EQ(ctx.device_type, nodes->ctx.device_type);
  }

  // Since DST nodes are included in SRC nodes, a common requirement is to fetch
  // the DST node features from the SRC nodes features. To avoid expensive sparse lookup,
  // the function assures that the DST nodes in both SRC and DST sets have the same ids.
  // As a result, given the node feature tensor ``X`` of type ``utype``,
  // the following code finds the corresponding DST node features of type ``vtype``:

  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();

  CHECK(rhs_nodes.size() == static_cast<size_t>(num_ntypes))
    << "rhs_nodes not given for every node type";

  std::vector<EdgeArray> edge_arrays(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t dsttype = src_dst_types.second;
    if (!aten::IsNullArray(rhs_nodes[dsttype])) {
      edge_arrays[etype] = graph->Edges(etype);
    }
  }

  // count lhs and rhs nodes
  std::vector<int64_t> maxNodesPerType(num_ntypes*2, 0);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    maxNodesPerType[ntype+num_ntypes] += rhs_nodes[ntype]->shape[0];

    if (generate_lhs_nodes) {
      if (include_rhs_in_lhs) {
        maxNodesPerType[ntype] += rhs_nodes[ntype]->shape[0];
      }
    } else {
      maxNodesPerType[ntype] += lhs_nodes[ntype]->shape[0];
    }
  }
  if (generate_lhs_nodes) {
    // we don't have lhs_nodes, see we need to count inbound edges to get an
    // upper bound
    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const dgl_type_t srctype = src_dst_types.first;
      if (edge_arrays[etype].src.defined()) {
        maxNodesPerType[srctype] += edge_arrays[etype].src->shape[0];
      }
    }
  }

  // gather lhs_nodes
  std::vector<IdArray> src_nodes(num_ntypes);
  if (generate_lhs_nodes) {
    std::vector<int64_t> src_node_offsets(num_ntypes, 0);
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      src_nodes[ntype] = NewIdArray(maxNodesPerType[ntype], ctx,
          sizeof(IdType)*8);
      if (include_rhs_in_lhs) {
        // place rhs nodes first
        device->CopyDataFromTo(rhs_nodes[ntype].Ptr<IdType>(), 0,
            src_nodes[ntype].Ptr<IdType>(), src_node_offsets[ntype],
            sizeof(IdType)*rhs_nodes[ntype]->shape[0],
            rhs_nodes[ntype]->ctx, src_nodes[ntype]->ctx,
            rhs_nodes[ntype]->dtype);
        src_node_offsets[ntype] += sizeof(IdType)*rhs_nodes[ntype]->shape[0];
      }
    }
    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const dgl_type_t srctype = src_dst_types.first;
      if (edge_arrays[etype].src.defined()) {
        device->CopyDataFromTo(
            edge_arrays[etype].src.Ptr<IdType>(), 0,
            src_nodes[srctype].Ptr<IdType>(),
            src_node_offsets[srctype],
            sizeof(IdType)*edge_arrays[etype].src->shape[0],
            rhs_nodes[srctype]->ctx,
            src_nodes[srctype]->ctx,
            rhs_nodes[srctype]->dtype);

        src_node_offsets[srctype] += sizeof(IdType)*edge_arrays[etype].src->shape[0];
      }
    }
  } else {
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      src_nodes[ntype] = lhs_nodes[ntype];
    }
  }

  // allocate space for map creation process
  DeviceNodeMapMaker<IdType> maker(maxNodesPerType);
  DeviceNodeMap<IdType> node_maps(maxNodesPerType, num_ntypes, ctx, stream);

  if (generate_lhs_nodes) {
    lhs_nodes.reserve(num_ntypes);
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      lhs_nodes.emplace_back(NewIdArray(
          maxNodesPerType[ntype], ctx, sizeof(IdType)*8));
    }
  }

  std::vector<int64_t> num_nodes_per_type(num_ntypes*2);
  // populate RHS nodes from what we already know
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    num_nodes_per_type[num_ntypes+ntype] = rhs_nodes[ntype]->shape[0];
  }

  // populate the mappings
  if (generate_lhs_nodes) {
    int64_t * count_lhs_device = static_cast<int64_t*>(
        device->AllocWorkspace(ctx, sizeof(int64_t)*num_ntypes*2));

    maker.Make(
        src_nodes,
        rhs_nodes,
        &node_maps,
        count_lhs_device,
        &lhs_nodes,
        stream);

    device->CopyDataFromTo(
        count_lhs_device, 0,
        num_nodes_per_type.data(), 0,
        sizeof(*num_nodes_per_type.data())*num_ntypes,
        ctx,
        DGLContext{kDLCPU, 0},
        DGLType{kDLInt, 64, 1});
    device->StreamSync(ctx, stream);

    // wait for the node counts to finish transferring
    device->FreeWorkspace(ctx, count_lhs_device);
  } else {
    maker.Make(
        lhs_nodes,
        rhs_nodes,
        &node_maps,
        stream);

    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      num_nodes_per_type[ntype] = lhs_nodes[ntype]->shape[0];
    }
  }

  std::vector<IdArray> induced_edges;
  induced_edges.reserve(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    if (edge_arrays[etype].id.defined()) {
      induced_edges.push_back(edge_arrays[etype].id);
    } else {
      induced_edges.push_back(
            aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx));
    }
  }

  // build metagraph -- small enough to be done on CPU
  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      num_ntypes * 2, etypes.src, new_dst);

  // allocate vector for graph relations while GPU is busy
  std::vector<HeteroGraphPtr> rel_graphs;
  rel_graphs.reserve(num_etypes);

  // map node numberings from global to local, and build pointer for CSR
  std::vector<IdArray> new_lhs;
  std::vector<IdArray> new_rhs;
  std::tie(new_lhs, new_rhs) = MapEdges(graph, edge_arrays, node_maps, stream);

  // resize lhs nodes
  if (generate_lhs_nodes) {
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      lhs_nodes[ntype]->shape[0] = num_nodes_per_type[ntype];
    }
  }

  // build the heterograph
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;

    if (rhs_nodes[dsttype]->shape[0] == 0) {
      // No rhs nodes are given for this edge type. Create an empty graph.
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_nodes[srctype]->shape[0], rhs_nodes[dsttype]->shape[0],
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx),
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx)));
    } else {
      rel_graphs.push_back(CreateFromCOO(
          2,
          lhs_nodes[srctype]->shape[0],
          rhs_nodes[dsttype]->shape[0],
          new_lhs[etype],
          new_rhs[etype]));
    }
  }

  HeteroGraphPtr new_graph = CreateHeteroGraph(
      new_meta_graph, rel_graphs, num_nodes_per_type);

  // return the new graph, the new src nodes, and new edges
  return std::make_tuple(new_graph, induced_edges);
}


template<typename IdType>
std::tuple<IdArray, IdArray,IdArray>
ToBlockGPUWithoutGraph(
    IdArray &lhs_nodes,
    IdArray &rhs_nodes, 
    IdArray &uni_nodes,
    bool include_rhs_in_lhs
    ) {
    cudaStream_t stream = runtime::getCurrentCUDAStream();
    auto ctx = lhs_nodes->ctx;
    auto device = runtime::DeviceAPI::Get(ctx);
    CHECK_EQ(ctx.device_type, kDLGPU);

    const int64_t num_etypes = 1;
    const int64_t num_ntypes = 1;

    std::vector<int64_t> maxNodesPerType(2, 0);
    
    // dst add --> src ,dst
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      maxNodesPerType[ntype+num_ntypes] += rhs_nodes->shape[0];
      if (include_rhs_in_lhs) {
        maxNodesPerType[ntype] += rhs_nodes->shape[0];
      }
    }

    // src add
    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      maxNodesPerType[etype] += lhs_nodes->shape[0];
    }

    // src list
    IdArray src_nodes =  NewIdArray(maxNodesPerType[0], ctx, sizeof(IdType)*8);
    int64_t src_node_offsets = 0;
    // copy dst --> src
    if (include_rhs_in_lhs) {
        device->CopyDataFromTo(rhs_nodes.Ptr<IdType>(), 0,
            src_nodes.Ptr<IdType>(), src_node_offsets,
            sizeof(IdType)*rhs_nodes->shape[0],
            rhs_nodes->ctx, src_nodes->ctx,
            rhs_nodes->dtype);
        src_node_offsets += sizeof(IdType)*rhs_nodes->shape[0];
    }
    // copy src --> src
    device->CopyDataFromTo(
      lhs_nodes.Ptr<IdType>(), 0,
      src_nodes.Ptr<IdType>(),
      src_node_offsets,
      sizeof(IdType)*lhs_nodes->shape[0],
      rhs_nodes->ctx,
      src_nodes->ctx,
      rhs_nodes->dtype);
    src_node_offsets += sizeof(IdType)*lhs_nodes->shape[0];

    DeviceNodeMapMaker<IdType> maker(maxNodesPerType);
    DeviceNodeMap<IdType> node_maps(maxNodesPerType, num_ntypes, ctx, stream);
    
    std::vector<int64_t> num_nodes_per_type(num_ntypes*2);
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      num_nodes_per_type[num_ntypes+ntype] = rhs_nodes->shape[0];
    }
    
    int64_t * count_lhs_device = static_cast<int64_t*>(
        device->AllocWorkspace(ctx, sizeof(int64_t)*num_ntypes*2));
    
    maker.Make( 
        src_nodes,
        rhs_nodes,
        &node_maps,
        count_lhs_device,
        uni_nodes,
        stream);
    
    device->CopyDataFromTo(
        count_lhs_device, 0,
        num_nodes_per_type.data(), 0,
        sizeof(*num_nodes_per_type.data())*num_ntypes,
        ctx,
        DGLContext{kDLCPU, 0},
        DGLType{kDLInt, 64, 1});
    device->StreamSync(ctx, stream);

    // wait for the node counts to finish transferring
    device->FreeWorkspace(ctx, count_lhs_device);
    
    uni_nodes->shape[0] = num_nodes_per_type[0];

    IdArray new_lhs;
    IdArray new_rhs;
    std::tie(new_lhs, new_rhs) = MapEdgesWithoutGraph(lhs_nodes, rhs_nodes, node_maps, stream);
    return std::make_tuple(new_lhs, new_rhs, uni_nodes);
}

template<typename IdType>
std::tuple<IdArray, IdArray,IdArray>
c_mapByNodeToEdge(
  IdArray &lhs_nodes,
  IdArray &rhs_nodes,
  IdArray &uni_nodes,
  IdArray &srcList,
  IdArray &dstList,
  bool include_rhs_in_lhs) {

    cudaStream_t stream = runtime::getCurrentCUDAStream();
    auto ctx = lhs_nodes->ctx;
    auto device = runtime::DeviceAPI::Get(ctx);
    CHECK_EQ(ctx.device_type, kDLGPU);

    const int64_t num_etypes = 1;
    const int64_t num_ntypes = 1;

    std::vector<int64_t> maxNodesPerType(2, 0);
    
    // dst add --> src ,dst
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      maxNodesPerType[ntype+num_ntypes] += rhs_nodes->shape[0];
      if (include_rhs_in_lhs) {
        maxNodesPerType[ntype] += rhs_nodes->shape[0];
      }
    }

    // src add
    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      maxNodesPerType[etype] += lhs_nodes->shape[0];
    }

    // src list
    IdArray src_nodes =  NewIdArray(maxNodesPerType[0], ctx, sizeof(IdType)*8);
    int64_t src_node_offsets = 0;
    // copy dst --> src
    if (include_rhs_in_lhs) {
        device->CopyDataFromTo(rhs_nodes.Ptr<IdType>(), 0,
            src_nodes.Ptr<IdType>(), src_node_offsets,
            sizeof(IdType)*rhs_nodes->shape[0],
            rhs_nodes->ctx, src_nodes->ctx,
            rhs_nodes->dtype);
        src_node_offsets += sizeof(IdType)*rhs_nodes->shape[0];
    }
    // copy src --> src
    device->CopyDataFromTo(
      lhs_nodes.Ptr<IdType>(), 0,
      src_nodes.Ptr<IdType>(),
      src_node_offsets,
      sizeof(IdType)*lhs_nodes->shape[0],
      rhs_nodes->ctx,
      src_nodes->ctx,
      rhs_nodes->dtype);
    src_node_offsets += sizeof(IdType)*lhs_nodes->shape[0];

    DeviceNodeMapMaker<IdType> maker(maxNodesPerType);
    DeviceNodeMap<IdType> node_maps(maxNodesPerType, num_ntypes, ctx, stream);
    
    std::vector<int64_t> num_nodes_per_type(num_ntypes*2);
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      num_nodes_per_type[num_ntypes+ntype] = rhs_nodes->shape[0];
    }
    
    int64_t * count_lhs_device = static_cast<int64_t*>(
        device->AllocWorkspace(ctx, sizeof(int64_t)*num_ntypes*2));
    
    maker.Make( 
        src_nodes,
        rhs_nodes,
        &node_maps,
        count_lhs_device,
        uni_nodes,
        stream);
    
    device->CopyDataFromTo(
        count_lhs_device, 0,
        num_nodes_per_type.data(), 0,
        sizeof(*num_nodes_per_type.data())*num_ntypes,
        ctx,
        DGLContext{kDLCPU, 0},
        DGLType{kDLInt, 64, 1});
    device->StreamSync(ctx, stream);

    // wait for the node counts to finish transferring
    device->FreeWorkspace(ctx, count_lhs_device);
    
    uni_nodes->shape[0] = num_nodes_per_type[0];

    IdArray new_lhs;
    IdArray new_rhs;
    std::tie(new_lhs, new_rhs) = MapEdgesWithoutGraph(srcList, dstList, node_maps, stream);
    return std::make_tuple(new_lhs, new_rhs, uni_nodes);
  }

void 
ToLoadGraphHalo(
  IdArray &indptr,
  IdArray &indices, 
  IdArray &edges,
  IdArray &bound,
  int gap
) {
  int32_t NUM = bound->shape[0] - 1;
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (NUM + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);
  
  unsigned long timeseed =
      std::chrono::system_clock::now().time_since_epoch().count();

  int32_t* in_ptr = static_cast<int32_t*>(indptr->data);
  int32_t* in_cols = static_cast<int32_t*>(indices->data);
  int32_t* in_edges = static_cast<int32_t*>(edges->data);
  int32_t* in_bound = static_cast<int32_t*>(bound->data);

  graph_halo_merge_kernel<blockSize, slice>
  <<<grid,block>>>(in_cols,in_ptr,in_edges,in_bound,NUM,gap,timeseed);
  cudaDeviceSynchronize();
}

}  // namespace

// Use explicit names to get around MSVC's broken mangling that thinks the following two
// functions are the same.
// Using template<> fails to export the symbols.
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
// ToBlock<kDLGPU, int32_t>
ToBlockGPU32(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs,
    std::vector<IdArray>* const lhs_nodes) {
  return ToBlockGPU<int32_t>(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

std::tuple<HeteroGraphPtr, std::vector<IdArray>>
// ToBlock<kDLGPU, int64_t>
ToBlockGPU64(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs,
    std::vector<IdArray>* const lhs_nodes) {
  return ToBlockGPU<int64_t>(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

std::tuple<IdArray, IdArray, IdArray>
// ToBlock<kDLGPU, int64_t>
ReMapIds(
    IdArray &lhs_nodes,
    IdArray &rhs_nodes,
    IdArray &uni_nodes,
    bool include_rhs_in_lhs) {
  return ToBlockGPUWithoutGraph<int32_t>(lhs_nodes,rhs_nodes,uni_nodes,include_rhs_in_lhs);
}

std::tuple<IdArray,IdArray,IdArray>
mapByNodeToEdge(
  IdArray &lhsNode,
  IdArray &rhsNode,
  IdArray &uniTable,
  IdArray &srcList,
  IdArray &dstList,
  bool include_rhs_in_lhs) {
    return c_mapByNodeToEdge<int32_t>(lhsNode,rhsNode,uniTable,srcList,dstList,include_rhs_in_lhs);
  }

void
c_loadGraphHalo(
  IdArray &indptr,
  IdArray &indices,
  IdArray &edges,
  IdArray &bound,
  int gap) {
    ToLoadGraphHalo(indptr,indices,edges,bound,gap);
}

void
c_FindNeighborByBfs(
  IdArray &nodeTable,
  IdArray &tmpTable,
  IdArray &srcList,
  IdArray &dstList,
  int64_t flag,
  bool acc) {

  int64_t NUM = srcList->shape[0];
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (NUM + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);
  
  int32_t* in_nodeTable = static_cast<int32_t*>(nodeTable->data);
  int32_t* in_tmpTable = static_cast<int32_t*>(tmpTable->data);
  int32_t* in_srcList = static_cast<int32_t*>(srcList->data);
  int32_t* in_dstList = static_cast<int32_t*>(dstList->data);

  ToUseBfsKernel<blockSize, slice>
  <<<grid,block>>>(in_nodeTable,in_tmpTable,in_srcList,in_dstList,NUM,flag,acc);
  cudaDeviceSynchronize();

  int64_t nodeNUM = nodeTable->shape[0];
  steps = (nodeNUM + slice - 1) / slice;
  dim3 grid_(steps);
  dim3 block_(blockSize);
  ToMergeTable<blockSize, slice>
  <<<grid_,block_>>>(in_nodeTable,in_tmpTable,nodeNUM,acc);
  cudaDeviceSynchronize();
}

void
c_FindNeigEdgeByBfs(
  IdArray &nodeTable,
  IdArray &tmpNodeTable,
  IdArray &edgeTable,
  IdArray &srcList,
  IdArray &dstList,
  int64_t offset,
  int32_t loopFlag) {
    int64_t NUM = srcList->shape[0];
    const int slice = 1024;
    const int blockSize = 256;
    int steps = (NUM + slice - 1) / slice;
    dim3 grid(steps);
    dim3 block(blockSize);
    
    int32_t* in_nodeTable = static_cast<int32_t*>(nodeTable->data);
    int32_t* in_tmpNodeTable = static_cast<int32_t*>(tmpNodeTable->data);
    int32_t* in_edgeTable = static_cast<int32_t*>(edgeTable->data);
    int32_t* in_srcList = static_cast<int32_t*>(srcList->data);
    int32_t* in_dstList = static_cast<int32_t*>(dstList->data);

    
    ToUseBfsWithEdgeKernel<blockSize, slice>
    <<<grid,block>>>(in_nodeTable,in_tmpNodeTable,in_edgeTable,in_srcList,in_dstList,NUM,offset,loopFlag);
    cudaDeviceSynchronize();

    int64_t nodeNUM = nodeTable->shape[0];
    steps = (nodeNUM + slice - 1) / slice;
    dim3 grid_(steps);
    dim3 block_(blockSize);
    ToMergeTable<blockSize, slice>
    <<<grid_,block_>>>(in_nodeTable,in_tmpNodeTable,nodeNUM,false);
    cudaDeviceSynchronize();
  }

void
c_maplocalIds(
  IdArray &nodeTable,
  IdArray &Gids,
  IdArray &Lids) {
    int64_t idsNUM = Gids->shape[0];
    int64_t TableNUM = nodeTable->shape[0];
    const int slice = 1024;
    const int blockSize = 256;
    int steps = (idsNUM + slice - 1) / slice;
    dim3 grid(steps);
    dim3 block(blockSize);
    
    int32_t* in_nodeTable = static_cast<int32_t*>(nodeTable->data);
    int32_t* in_gids = static_cast<int32_t*>(Gids->data);
    int32_t* in_lids = static_cast<int32_t*>(Lids->data);

    
    ToMapLocalIdKernel<blockSize, slice>
    <<<grid,block>>>(in_nodeTable,in_gids,in_lids,TableNUM,idsNUM);
    cudaDeviceSynchronize();
  }

void 
c_findSameNode (
  IdArray &tensor1,
  IdArray &tensor2,
  IdArray &indexTable1,
  IdArray &indexTable2
) {
  int64_t t1NUM = tensor1->shape[0];
  int64_t t2NUM = tensor2->shape[0];
  int64_t idsNUM = t1NUM > t2NUM ? t1NUM : t2NUM;
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (idsNUM + slice - 1) / slice;
  dim3 grid(steps,2);
  dim3 block(blockSize);

  int32_t* in_t1 = static_cast<int32_t*>(tensor1->data);
  int32_t* in_t2 = static_cast<int32_t*>(tensor2->data);
  int32_t* in_table1 = static_cast<int32_t*>(indexTable1->data);
  int32_t* in_table2 = static_cast<int32_t*>(indexTable2->data);


  ToFindSameNodeKernel<blockSize, slice>
    <<<grid,block>>>(in_t1,in_t2,in_table1,in_table2,t1NUM,t2NUM);
  cudaDeviceSynchronize();
}

void
c_SumDegree(
  IdArray &InNodeTabel,
  IdArray &OutnodeTabel,
  IdArray &srcList,
  IdArray &dstList){
  int64_t edgeNUM = srcList->shape[0];
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (edgeNUM + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);

  int32_t* in_nodeTabel = static_cast<int32_t*>(InNodeTabel->data);
  int32_t* Out_nodeTabel = static_cast<int32_t*>(OutnodeTabel->data);
  int32_t* in_srcList = static_cast<int32_t*>(srcList->data);
  int32_t* in_dstList= static_cast<int32_t*>(dstList->data);


  SumDegreeKernel<blockSize, slice>
    <<<grid,block>>>(in_nodeTabel,Out_nodeTabel,in_srcList,in_dstList,edgeNUM);
  cudaDeviceSynchronize();
}



void
c_calculateP(
  IdArray &DegreeTabel,
  IdArray &PTabel,
  IdArray &srcList,
  IdArray &dstList,
  int64_t fanout){
  
  int64_t edgeNUM = srcList->shape[0];
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (edgeNUM + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);

  int32_t* in_DegreeTabel = static_cast<int32_t*>(DegreeTabel->data);
  int32_t* in_PTabel = static_cast<int32_t*>(PTabel->data);
  int32_t* in_srcList = static_cast<int32_t*>(srcList->data);
  int32_t* in_dstList= static_cast<int32_t*>(dstList->data);


  calculatePKernel<blockSize, slice>
    <<<grid,block>>>(in_DegreeTabel,in_PTabel,in_srcList,in_dstList,edgeNUM,fanout);
  cudaDeviceSynchronize();
}


void
c_PPR(
  IdArray &src,
  IdArray &dst,
  IdArray &edgeTable,
  IdArray &degreeTable,
  IdArray &nodeValue,
  IdArray &nodeInfo,
  IdArray &tmpNodeValue,
  IdArray &tmpNodeInfo) {

  int64_t edgeNUM = src->shape[0];
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (edgeNUM + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);

  int32_t* in_src = static_cast<int32_t*>(src->data);
  int32_t* in_dst = static_cast<int32_t*>(dst->data);
  int32_t* in_edgeTable = static_cast<int32_t*>(edgeTable->data);
  int32_t* in_degreeTable = static_cast<int32_t*>(degreeTable->data);
  int32_t* in_nodeValue = static_cast<int32_t*>(nodeValue->data);
  int32_t* in_nodeInfo= static_cast<int32_t*>(nodeInfo->data);
  int32_t* in_tmpNodeValue = static_cast<int32_t*>(tmpNodeValue->data);
  int32_t* in_tmpNodeInfo= static_cast<int32_t*>(tmpNodeInfo->data);
  

  PPRkernel<blockSize, slice>
    <<<grid,block>>>(in_src,in_dst,in_edgeTable,in_degreeTable,in_nodeValue,in_nodeInfo,in_tmpNodeValue,in_tmpNodeInfo,edgeNUM);
  cudaDeviceSynchronize();

  int64_t nodeNUM = nodeValue->shape[0];
  steps = (nodeNUM + slice - 1) / slice;
  dim3 _grid(steps);
  dim3 _block(blockSize);
  
  ToMergeTable<blockSize, slice>
    <<<_grid,_block>>>(in_nodeValue,in_tmpNodeValue,nodeNUM,true);
  cudaDeviceSynchronize();
  ToMergeTable<blockSize, slice>
    <<<_grid,_block>>>(in_nodeInfo,in_tmpNodeInfo,nodeNUM,false);
  cudaDeviceSynchronize();

}


}  // namespace transform
}  // namespace dgl
