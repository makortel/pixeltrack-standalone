#include "CUDACore/ConcreteContext.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/Product.h"
#include "CUDACore/StreamCache.h"

#include "chooseDevice.h"

namespace cms::cuda {
  namespace impl {
    ContextBase::ContextBase(edm::StreamID streamID) {
      auto device = cms::cuda::chooseDevice(streamID);
      cudaCheck(cudaSetDevice(device));
      setDeviceStream(device, cms::cuda::getStreamCache().get());
    }

    ContextBase::ContextBase(const ProductBase& data) {
      auto const device = data.device();
      cudaCheck(cudaSetDevice(device));
      if (data.mayReuseStream()) {
        setDeviceStream(device, data.streamPtr());
      } else {
        setDeviceStream(device, cms::cuda::getStreamCache().get());
      }
    }

    ////
    void ContextGetterBase::synchronizeStreams(int dataDevice,
                                               cudaStream_t dataStream,
                                               bool available,
                                               cudaEvent_t dataEvent) {
      if (dataDevice != device()) {
        // Eventually replace with prefetch to current device (assuming unified memory works)
        // If we won't go to unified memory, need to figure out something else...
        throw std::runtime_error("Handling data from multiple devices is not yet supported");
      }

      if (dataStream != stream()) {
        // Different streams, need to synchronize
        if (not available) {
          // Event not yet occurred, so need to add synchronization
          // here. Sychronization is done by making the CUDA stream to
          // wait for an event, so all subsequent work in the stream
          // will run only after the event has "occurred" (i.e. data
          // product became available).
          cudaCheck(cudaStreamWaitEvent(stream(), dataEvent, 0), "Failed to make a stream to wait for an event");
        }
      }
    }

  }
}
