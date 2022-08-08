#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>


std::vector<torch::Tensor> cuda_ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj, 
    torch::Tensor kk,
    int t0, int t1, int iterations);


torch::Tensor cuda_reproject(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj, 
    torch::Tensor kk);

std::vector<torch::Tensor> ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj, 
    torch::Tensor kk,
    int t0, int t1, int iterations) {
  return cuda_ba(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations);
}


torch::Tensor reproject(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj, 
    torch::Tensor kk) {
  return cuda_reproject(poses, patches, intrinsics, ii, jj, kk);
}

// std::vector<torch::Tensor> neighbors(torch::Tensor ii, torch::Tensor jj)
// {
//   ii = ii.to(torch::kCPU);
//   jj = jj.to(torch::kCPU);
//   auto ii_data = ii.accessor<long,1>();
//   auto jj_data = jj.accessor<long,1>();

//   std::unordered_map<long, std::vector<long>> graph;
//   std::unordered_map<long, std::vector<long>> index;
//   for (int i=0; i < ii.size(0); i++) {
//     const long ix = ii_data[i];
//     const long jx = jj_data[i];
//     if (graph.find(ix) == graph.end()) {
//       graph[ix] = std::vector<long>();
//       index[ix] = std::vector<long>();
//     }
//     graph[ix].push_back(jx);
//     index[ix].push_back( i);
//   }

//   auto opts = torch::TensorOptions().dtype(torch::kInt64);
//   torch::Tensor ix = torch::empty({ii.size(0)}, opts);
//   torch::Tensor jx = torch::empty({jj.size(0)}, opts); 

//   auto ix_data = ix.accessor<long,1>();
//   auto jx_data = jx.accessor<long,1>();

//   for (std::pair<long, std::vector<long>> element : graph) {
//     std::vector<long>& v = graph[element.first];
//     std::vector<long>& idx = index[element.first];

//     std::stable_sort(idx.begin(), idx.end(),
//        [&v](size_t i, size_t j) {return v[i] < v[j];});

//     ix_data[idx.front()] = -1;
//     jx_data[idx.back()]  = -1;

//     for (int i=0; i < idx.size(); i++) {
//       ix_data[idx[i]] = (i > 0) ? idx[i-1] : -1;
//       jx_data[idx[i]] = (i < idx.size() - 1) ? idx[i+1] : -1;
//     }
//   }

//   ix = ix.to(torch::kCUDA);
//   jx = jx.to(torch::kCUDA);

//   return {ix, jx};
// }


std::vector<torch::Tensor> neighbors(torch::Tensor ii, torch::Tensor jj)
{

  auto tup = torch::_unique(ii, true, true);
  torch::Tensor uniq = std::get<0>(tup).to(torch::kCPU);
  torch::Tensor perm = std::get<1>(tup).to(torch::kCPU);

  jj = jj.to(torch::kCPU);
  auto jj_accessor = jj.accessor<long,1>();

  auto perm_accessor = perm.accessor<long,1>();
  std::vector<std::vector<long>> index(uniq.size(0));
  for (int i=0; i < ii.size(0); i++) {
    index[perm_accessor[i]].push_back(i);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor ix = torch::empty({ii.size(0)}, opts);
  torch::Tensor jx = torch::empty({ii.size(0)}, opts); 

  auto ix_accessor = ix.accessor<long,1>();
  auto jx_accessor = jx.accessor<long,1>();

  for (int i=0; i<uniq.size(0); i++) {
    std::vector<long>& idx = index[i];
    std::stable_sort(idx.begin(), idx.end(),
       [&jj_accessor](size_t i, size_t j) {return jj_accessor[i] < jj_accessor[j];});

    for (int i=0; i < idx.size(); i++) {
      ix_accessor[idx[i]] = (i > 0) ? idx[i-1] : -1;
      jx_accessor[idx[i]] = (i < idx.size() - 1) ? idx[i+1] : -1;
    }
  }

  // for (int i=0; i<ii.size(0); i++) {
  //   std::cout << jj_accessor[i] << " ";
  //   if (ix_accessor[i] >= 0) std::cout << jj_accessor[ix_accessor[i]] << " ";
  //   if (jx_accessor[i] >= 0) std::cout << jj_accessor[jx_accessor[i]] << " ";
  //   std::cout << std::endl;
  // }

  ix = ix.to(torch::kCUDA);
  jx = jx.to(torch::kCUDA);

  return {ix, jx};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ba, "BA forward operator");
  m.def("neighbors", &neighbors, "temporal neighboor indicies");
  m.def("reproject", &reproject, "temporal neighboor indicies");

}