#include "CUDACore/ConcreteContext.h"

#include "chooseDevice.h"

namespace cms::cuda {
  namespace impl {
    ContextBase::ContextBase(edm::StreamID streamID) : currentDevice_(chooseDevice(streamID)) {
      cudaCheck(cudaSetDevice(currentDevice_));
      stream_ = getStreamCache().get();
    }

    ScopedContextBase::ScopedContextBase(const ProductBase& data) : currentDevice_(data.device()) {
      cudaCheck(cudaSetDevice(currentDevice_));
      if (data.mayReuseStream()) {
        stream_ = data.streamPtr();
      } else {
        stream_ = getStreamCache().get();
      }
    }

  }
}
