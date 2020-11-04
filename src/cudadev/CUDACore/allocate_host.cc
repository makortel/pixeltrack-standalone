#include <limits>

#include "CUDACore/allocate_host.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingHostAllocator.h"

namespace cms::cuda {
  void *allocate_host(size_t nbytes, cudaStream_t stream) {
    if constexpr (allocator::useCaching) {
      return allocator::getCachingHostAllocator().allocate(allocator::HostTraits::kHostDevice, nbytes, stream);
    } else {
      void *ptr = nullptr;
      cudaCheck(cudaMallocHost(&ptr, nbytes));
      return ptr;
    }
  }

  void free_host(void *ptr) {
    if constexpr (allocator::useCaching) {
      allocator::getCachingHostAllocator().free(allocator::HostTraits::kHostDevice, ptr);
    } else {
      cudaCheck(cudaFreeHost(ptr));
    }
  }

}  // namespace cms::cuda
