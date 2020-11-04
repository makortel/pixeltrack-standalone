#ifndef HeterogeneousCore_CUDACore_src_getCachingHostAllocator
#define HeterogeneousCore_CUDACore_src_getCachingHostAllocator

#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace cms::cuda::allocator {
  struct HostTraits {
    using DeviceType = int;
    using QueueType = cudaStream_t;
    using EventType = cudaEvent_t;
    struct Dummy {};

    static constexpr DeviceType kInvalidDevice = -1;
    static constexpr DeviceType kHostDevice = 0;

    static DeviceType currentDevice() { return cms::cuda::currentDevice(); }

    static Dummy setDevice(DeviceType device) { return {}; }

    static bool canReuseInDevice(DeviceType a, DeviceType b) {
      // Pinned host memory can be reused in any device, but in case of
      // changes the event must be re-created
      return true;
    }

    static bool canReuseInQueue(QueueType a, QueueType b) {
      // For pinned host memory a freed block without completed event
      // can not be re-used even for operations in the same queue
      return false;
    }

    static bool eventWorkHasCompleted(EventType e) { return cms::cuda::eventWorkHasCompleted(e); }

    static EventType createEvent() {
      EventType e;
      cudaCheck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      return e;
    }

    static void destroyEvent(EventType e) { cudaCheck(cudaEventDestroy(e)); }

    static EventType recreateEvent(EventType e, DeviceType prev, DeviceType next) {
      cudaCheck(cudaSetDevice(prev));
      destroyEvent(e);
      cudaCheck(cudaSetDevice(next));
      return createEvent();
    }

    static EventType recordEvent(EventType e, QueueType queue) {
      cudaCheck(cudaEventRecord(e, queue));
      return e;
    }

    static std::ostream& printDevice(std::ostream& os, DeviceType dev) {
      os << "Host";
      return os;
    }

    static void* allocate(size_t bytes) {
      void* ptr;
      cudaCheck(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
      return ptr;
    }

    static void* tryAllocate(size_t bytes) {
      void* ptr;
      auto error = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
      if (error == cudaErrorMemoryAllocation) {
        return nullptr;
      }
      cudaCheck(error);
      return ptr;
    }

    static void free(void* ptr) { cudaCheck(cudaFreeHost(ptr)); }
  };

  using CachingHostAllocator = GenericCachingAllocator<HostTraits>;

  inline CachingHostAllocator& getCachingHostAllocator() {
    // the public interface is thread safe
    static CachingHostAllocator allocator{binGrowth, minBin, maxBin, minCachedBytes(), debug};
    return allocator;
  }
}  // namespace cms::cuda::allocator

#endif
