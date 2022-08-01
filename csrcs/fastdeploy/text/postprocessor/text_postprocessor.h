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

#pragma once

#include <vector>
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/text/common/result.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace text {

class TextPostprocessor {
 public:
  virtual bool Decode(const std::vector<FDTensor>& model_result,
                      TextResult* decoded_result) const;
  virtual bool DecodeBatch(const std::vector<FDTensor>& model_result,
                           BatchTextResult* decoded_result) const;
};

}  // namespace text
}  // namespace fastdeploy
