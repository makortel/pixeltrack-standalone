#include "CUDACore/Context.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"
#include "getCachingHostAllocator.h"

namespace cms::cuda {
  namespace impl {
    void DeviceDeleter::operator()(void *ptr) {
      if constexpr (allocator::useCaching) {
        allocator::getCachingDeviceAllocator().free(device_, ptr);
      } else {
        ScopedSetDevice setDeviceForThisScope(device_);
        cudaCheck(cudaFree(ptr));
      }
    }

    void HostDeleter::operator()(void *ptr) {
      if constexpr (allocator::useCaching) {
        allocator::getCachingHostAllocator().free(allocator::HostTraits::kHostDevice, ptr);
      } else {
        cudaCheck(cudaFreeHost(ptr));
      }
    }
  }  // namespace impl

  void *Context::allocate_device_impl(size_t bytes) {
    if constexpr (allocator::useCaching) {
      return allocator::getCachingDeviceAllocator().allocate(device(), bytes, stream());
    } else {
      void *ptr = nullptr;
      ScopedSetDevice setDeviceForThisScope(device());
      cudaCheck(cudaMalloc(&ptr, bytes));
      return ptr;
    }
  }

  void *Context::allocate_host_impl(size_t bytes) {
    if constexpr (allocator::useCaching) {
      return allocator::getCachingHostAllocator().allocate(allocator::HostTraits::kHostDevice, bytes, stream());
    } else {
      void *ptr = nullptr;
      cudaCheck(cudaMallocHost(&ptr, bytes));
      return ptr;
    }
  }
}  // namespace cms::cuda
