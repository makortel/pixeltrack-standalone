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
  }
}
