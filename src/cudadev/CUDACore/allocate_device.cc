#include <limits>

#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace cms::cuda {
  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    if constexpr (allocator::useCaching) {
      return allocator::getCachingDeviceAllocator().allocate(dev, nbytes, stream);
    } else {
      void *ptr = nullptr;
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(cudaMalloc(&ptr, nbytes));
      return ptr;
    }
  }

  void free_device(int device, void *ptr) {
    if constexpr (allocator::useCaching) {
      allocator::getCachingDeviceAllocator().free(device, ptr);
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda
