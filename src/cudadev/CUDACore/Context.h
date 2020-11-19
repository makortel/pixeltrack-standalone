#ifndef CUDACore_Context_h
#define CUDACore_Context_h

#include <memory>
#include <tuple>

#include "CUDACore/SharedStreamPtr.h"

namespace cms {
  namespace cuda {
    namespace impl {
      class DeviceDeleter {
      public:
        DeviceDeleter() = default;  // for edm::Wrapper
        DeviceDeleter(int device) : device_{device} {}

        void operator()(void* ptr);

      private:
        int device_ = -1;
      };

      class HostDeleter {
      public:
        void operator()(void* ptr);
      };
    }  // namespace impl

    class Context {
    public:
      int device() const { return currentDevice_; }

      // cudaStream_t is a pointer to a thread-safe object, for which a
      // mutable access is needed even if the ScopedContext itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      cudaStream_t stream() const { return stream_.get(); }

      // These are intended for make_device_unique and make_host_unique implementations
      auto allocate_device(size_t bytes) {
        return std::unique_ptr<void, impl::DeviceDeleter>(allocate_device_impl(bytes), impl::DeviceDeleter(device()));
      }
      auto allocate_host(size_t bytes) {
        return std::unique_ptr<void, impl::HostDeleter>(allocate_host_impl(bytes), impl::HostDeleter());
      }

    protected:
      explicit Context(std::tuple<SharedStreamPtr, int> streamDevice);

    private:
      void* allocate_device_impl(size_t bytes);
      void* allocate_host_impl(size_t bytes);

      int currentDevice_;
      SharedStreamPtr stream_;
    };

  }  // namespace cuda
}  // namespace cms

#endif
