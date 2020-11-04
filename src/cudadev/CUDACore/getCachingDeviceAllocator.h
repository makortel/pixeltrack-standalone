#ifndef HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator
#define HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator

#include "CUDACore/cudaCheck.h"
#include "CUDACore/currentDevice.h";
#include "CUDACore/deviceCount.h"
#include "CUDACore/eventWorkHasCompleted.h"
#include "CUDACore/GenericCachingAllocator.h"
#include "CUDACore/ScopedSetDevice.h"

namespace cms::cuda::allocator {
  // Use caching or not
  constexpr bool useCaching = true;
  // Growth factor (bin_growth in cub::CachingDeviceAllocator
  constexpr unsigned int binGrowth = 8;
  // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CacingDeviceAllocator
  constexpr unsigned int minBin = 1;
  // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator). Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
  constexpr unsigned int maxBin = 10;
  // Total storage for the allocator. 0 means no limit.
  constexpr size_t maxCachedBytes = 0;
  // Fraction of total device memory taken for the allocator. In case there are multiple devices with different amounts of memory, the smallest of them is taken. If maxCachedBytes is non-zero, the smallest of them is taken.
  constexpr double maxCachedFraction = 0.8;
  constexpr bool debug = false;

  inline size_t minCachedBytes() {
    size_t ret = std::numeric_limits<size_t>::max();
    int currentDevice;
    cudaCheck(cudaGetDevice(&currentDevice));
    const int numberOfDevices = deviceCount();
    for (int i = 0; i < numberOfDevices; ++i) {
      size_t freeMemory, totalMemory;
      cudaCheck(cudaSetDevice(i));
      cudaCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
      ret = std::min(ret, static_cast<size_t>(maxCachedFraction * freeMemory));
    }
    cudaCheck(cudaSetDevice(currentDevice));
    if (maxCachedBytes > 0) {
      ret = std::min(ret, maxCachedBytes);
    }
    return ret;
  }

  struct DeviceTraits {
    using DeviceType = int;
    using QueueType = cudaStream_t;
    using EventType = cudaEvent_t;

    static constexpr DeviceType kInvalidDevice = -1;

    static DeviceType currentDevice() { return cms::cuda::currentDevice(); }

    static cms::cuda::ScopedSetDevice setDevice(DeviceType device) { return cms::cuda::ScopedSetDevice(device); }

    static bool canReuseInDevice(DeviceType a, DeviceType b) { return a == b; }

    static bool canReuseInQueue(QueueType a, QueueType b) { return a == b; }

    static bool eventWorkHasCompleted(EventType e) { return cms::cuda::eventWorkHasCompleted(e); }

    static EventType createEvent() {
      EventType e;
      cudaCheck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      return e;
    }

    static void destroyEvent(EventType e) { cudaCheck(cudaEventDestroy(e)); }

    static EventType recreateEvent(EventType e, DeviceType prev, DeviceType next) {
      throw std::runtime_error("CUDADeviceTraits::recreateEvent() should never be called");
    }

    static void recordEvent(EventType e, QueueType queue) { cudaCheck(cudaEventRecord(e, queue)); }

    static std::ostream& printDevice(std::ostream& os, DeviceType device) {
      os << "Device " << device;
      return os;
    }

    static void* allocate(size_t bytes) {
      void* ptr;
      cudaCheck(cudaMalloc(&ptr, bytes));
      return ptr;
    }

    static void* tryAllocate(size_t bytes) {
      void* ptr;
      auto error = cudaMalloc(&ptr, bytes);
      if (error == cudaErrorMemoryAllocation) {
        return nullptr;
      }
      cudaCheck(error);
      return ptr;
    }

    static void free(void* ptr) { cudaCheck(cudaFree(ptr)); }
  };

  using CachingDeviceAllocator = GenericCachingAllocator<DeviceTraits>;

  inline CachingDeviceAllocator& getCachingDeviceAllocator() {
    // the public interface is thread safe
    static CachingDeviceAllocator allocator{binGrowth, minBin, maxBin, minCachedBytes(), debug};
    return allocator;
  }
}  // namespace cms::cuda::allocator

#endif
