#ifndef HeterogeneousCore_CUDAUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_device_unique_ptr_h

#include <memory>
#include <functional>

#include "CUDACore/allocate_device.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/Context.h"

namespace cms {
  namespace cuda {
    namespace device {
      namespace impl {
        using DeviceDeleter = cms::cuda::impl::DeviceDeleter;
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::DeviceDeleter>;

      namespace impl {
        template <typename T>
        struct make_device_unique_selector {
          using non_array = cms::cuda::device::unique_ptr<T>;
        };
        template <typename T>
        struct make_device_unique_selector<T[]> {
          using unbounded_array = cms::cuda::device::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_device_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace device

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique(cudaStream_t stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      int dev = currentDevice();
      void *mem = allocate_device(dev, sizeof(T), stream);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{dev}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique(size_t n,
                                                                                              cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      int dev = currentDevice();
      void *mem = allocate_device(dev, n * sizeof(element_type), stream);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{dev}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique(Args &&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique_uninitialized(
        cudaStream_t stream) {
      int dev = currentDevice();
      void *mem = allocate_device(dev, sizeof(T), stream);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{dev}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique_uninitialized(
        size_t n, cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      int dev = currentDevice();
      void *mem = allocate_device(dev, n * sizeof(element_type), stream);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{dev}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique_uninitialized(Args &&...) =
        delete;

    // With Context
    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique(Context &ctx) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      int dev = currentDevice();
      auto mem = ctx.allocate_device(sizeof(T));
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem.release()),
                                                                              mem.get_deleter()};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique(size_t n, Context &ctx) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      auto mem = ctx.allocate_device(n * sizeof(element_type));
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem.release()), mem.get_deleter()};
    }

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique_uninitialized(Context &ctx) {
      auto mem = ctx.allocate_device(sizeof(T));
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem.release()),
                                                                              mem.get_deleter()};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique_uninitialized(
        size_t n, Context &ctx) {
      using element_type = typename std::remove_extent<T>::type;
      auto mem = ctx.allocate_device(n * sizeof(element_type));
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem.release()), mem.get_deleter()};
    }
  }  // namespace cuda
}  // namespace cms

#endif
