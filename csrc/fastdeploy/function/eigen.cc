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

#include "fastdeploy/function/eigen.h"
#include "fastdeploy/utils/unique_ptr.h"
#ifdef WITH_GPU
#include <cuda_runtime_api.h>
#endif
#include <mutex>
#include <unordered_set>

#if !defined(_WIN32)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
// there is no equivalent intrinsics in msvc.
#define UNLIKELY(condition) (condition)
#endif

#if !defined(_WIN32)
#define LIKELY(condition) __builtin_expect(static_cast<bool>(condition), 1)
#else
// there is no equivalent intrinsics in msvc.
#define LIKELY(condition) (condition)
#endif

namespace fastdeploy {

std::shared_ptr<EigenDeviceWrapper> EigenDeviceWrapper::instance_ = nullptr;

std::shared_ptr<EigenDeviceWrapper> EigenDeviceWrapper::GetInstance() {
  if (instance_ == nullptr) {
    instance_ = std::make_shared<EigenDeviceWrapper>();
    instance_->device_ = utils::make_unique<Eigen::DefaultDevice>();
  }
  return instance_;
}

const Eigen::DefaultDevice* EigenDeviceWrapper::GetDevice() const {
  return device_.get();
}

#ifdef WITH_GPU

class EigenGpuStreamDevice : public Eigen::StreamInterface {
 public:
  EigenGpuStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenGpuStreamDevice() override {}

  void Reinitialize(cudaStream_t cuda_stream, int device_id) {
    stream_ = cuda_stream;
    device_prop_ = &Eigen::m_deviceProperties[device_id];
  }

  const cudaStream_t& stream() const override { return stream_; }

  const cudaDeviceProp& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    if (UNLIKELY(num_bytes == 0)) {
      return nullptr;
    }
    void* ptr;
    cudaMalloc(&ptr, num_bytes);
    return ptr;
  }

  void deallocate(void* buffer) const override {
    if (LIKELY(buffer)) {
      std::lock_guard<std::mutex> lock(mtx_);
      cudaFree(buffer);
    }
  }

  void* scratchpad() const override {
    if (scratch_ == NULL) {
      scratch_ = allocate(Eigen::kGpuScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == NULL) {
      char* scratch = static_cast<char*>(scratchpad()) + Eigen::kGpuScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream_);
    }
    return semaphore_;
  }

 private:
  cudaStream_t stream_{};              // not owned;
  const cudaDeviceProp* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
  mutable std::mutex mtx_;  // to protect allocations_
};

std::shared_ptr<EigenGpuDeviceWrapper> EigenGpuDeviceWrapper::instance_ =
    nullptr;

std::shared_ptr<EigenGpuDeviceWrapper> EigenGpuDeviceWrapper::GetInstance() {
  if (instance_ == nullptr) {
    instance_ = std::make_shared<EigenGpuDeviceWrapper>();
    instance_->eigen_stream_.reset(new EigenGpuStreamDevice());
    instance_->device_ =
        utils::make_unique<Eigen::GpuDevice>(instance_->eigen_stream_.get());
  }
  return instance_;
}

const Eigen::GpuDevice* EigenGpuDeviceWrapper::GetDevice() const {
  return device_.get();
}
#endif

}  // namespace fastdeploy
