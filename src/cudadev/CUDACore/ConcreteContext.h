#ifndef CUDACore_ConcreteContext_h
#define CUDACore_ConcreteContext_h

#include "CUDACore/Context.h"
#include "CUDACore/Product.h"
#include "Framework/Event.h"

namespace cms::cuda {
  namespace impl {
    class ContextBase : public Context {
    protected:
      // The constructors set the current device, but the device
      // is not set back to the previous value at the destructor. This
      // should be sufficient (and tiny bit faster) as all CUDA API
      // functions relying on the current device should be called from
      // the scope where this context is. The current device doesn't
      // really matter between modules (or across TBB tasks).
      explicit ContextBase(edm::StreamID streamID);

      explicit ContextBase(const ProductBase& data);
    };

    class ContextGetterBase : public ContextBase {
    public:
      template <typename T>
      const T& get(const Product<T>& data) {
        synchronizeStreams(data.device(), data.stream(), data.isAvailable(), data.event());
        return data.data_;
      }

      template <typename T>
      const T& get(const edm::Event& iEvent, edm::EDGetTokenT<Product<T>> token) {
        return get(iEvent.get(token));
      }

    protected:
      template <typename... Args>
      ContextGetterBase(Args&&... args) : ContextBase(std::forward<Args>(args)...) {}

      void synchronizeStreams(int dataDevice, cudaStream_t dataStream, bool available, cudaEvent_t dataEvent);
    };

  }
}

#endif
