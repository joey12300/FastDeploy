// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/function/quantile.h"
#include "fastdeploy/function/elementwise.h"
#include "fastdeploy/function/isfinite.h"
#include "fastdeploy/function/reduce.h"
#include "fastdeploy/function/sort.h"
#include "fastdeploy/function/transpose.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace fastdeploy {
namespace function {

template <typename T>
void QuantileKernel(const FDTensor& x, const std::vector<double>& q,
                    const std::vector<int>& axis, FDTensor* out) {
  FDASSERT(q.size() > 0, "q should not be empty.");
  FDASSERT(axis.size() > 0, "axis should not be empty.");
  std::vector<int64_t> axis_src;
  std::vector<int64_t> out_shape = x.Shape();
  int64_t rank = x.Shape().size();
  for (auto axis_single : axis) {
    FDASSERT(axis_single >= -rank && axis_single < rank,
             "The axis is expected to be in range of [%d, %d), but got %d",
             -rank, rank, axis_single);
    if (axis_single < 0) {
      axis_single += rank;
    }
    axis_src.push_back(axis_single);
    out_shape[axis_single] = 1;
  }
  out->Allocate(out_shape, x.Dtype());
  std::vector<int64_t> axis_dst;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axis_src.begin(), axis_src.end(), i) == axis_src.end()) {
      axis_dst.push_back(i);
    }
  }
  axis_dst.insert(axis_dst.end(), axis_src.begin(), axis_src.end());
  FDTensor y;
  Transpose(x, &y, axis_dst);
  std::vector<int64_t> y_shape(rank - axis_src.size(), 0);
  y_shape.push_back(-1);
  y.Reshape({y_shape});

  int64_t target_axis = rank - 1;
  FDTensor mask, valid_counts, mask_any;
  IsNan(y, &mask, FDDataType::FP64);
  Min(mask, &mask_any, {target_axis}, true);
  double* mask_data = reinterpret_cast<double*>(mask.Data());
  std::transform(mask_data, mask_data + mask.Numel(), mask_data,
                 [](const double& val) {
                   if (std::abs(val) < 1e-8) {
                     return 1;
                   }
                   return 0;
                 });
  Sum(mask, &valid_counts, {target_axis}, true);

  FDTensor one_tensor(static_cast<double>(1.0));

  std::vector<FDTensor> indices;
  FDTensor last_index(static_cast<double>(x.Shape()[target_axis]));
  for (auto q_num : q) {
    FDASSERT(q_num >= 0 && q_num <= 1, "q should be in range [0, 1]");
    FDTensor q_tensor, index;
    q_tensor.Allocate({1}, FDDataType::FP64);
    (reinterpret_cast<double*>(q_tensor.Data()))[0] = q_num;
    index = q_tensor * (valid_counts - one_tensor);
    index = mask_any * last_index + (one_tensor - mask_any) * index;
    indices.push_back(index);
  }

  std::vector<FDTensor> output;
  FDTensor sorted_tensor, sorted_indices_tensor;
  Sort(y, &sorted_tensor, &sorted_indices_tensor, target_axis);

  for (auto&& index : indices) {
  }
}

void Quantile(const FDTensor& x, const std::vector<double>& q,
              const std::vector<int>& axis, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "QuantileKernel",
                       ([&] { QuantileKernel<data_t>(x, q, axis, out); }));
}

}  // namespace function
}  // namespace fastdeploy