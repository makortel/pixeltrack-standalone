#include "CUDACore/ConcreteContext.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/Product.h"
#include "CUDACore/StreamCache.h"

#include "chooseDevice.h"

namespace {
  std::tuple<cms::cuda::SharedStreamPtr, int> getStream(edm::StreamID streamID) {
    auto device = cms::cuda::chooseDevice(streamID);
    cudaCheck(cudaSetDevice(device));
    return std::tuple(cms::cuda::getStreamCache().get(), device);
  }

  std::tuple<cms::cuda::SharedStreamPtr, int> getStream(cms::cuda::ProductBase const& data) {
    if (data.mayReuseStream()) {
      return std::tuple(data.streamPtr(), data.device());
    } else {
      cudaCheck(cudaSetDevice(data.device));
      return std::tuple(cms::cuda::getStreamCache().get(), data.device());
    }
  }
}

namespace cms::cuda {
  namespace impl {
    ContextBase::ContextBase(edm::StreamID streamID) : Context(getStream(streamID)) {}

    ContextBase::ContextBase(const ProductBase& data) : Context(getStream(data)) {}
  }
}
