#include "CUDADataFormats/SiPixelDigiErrorsCUDA.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/memsetAsync.h"

#include <cassert>

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cudaStream_t stream)
    : formatterErrors_h(std::move(errors)), maxFedWords_(maxFedWords) {
  error_d = cms::cuda::make_managed_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
  data_d = cms::cuda::make_managed_unique<PixelErrorCompact[]>(maxFedWords, stream);

  std::memset(data_d.get(), 0, maxFedWords * sizeof(PixelErrorCompact));
  GPU::make_SimpleVector(error_d.get(), maxFedWords, data_d.get());
  assert(error_d->empty());
  assert(error_d->capacity() == static_cast<int>(maxFedWords));

  auto device = cms::cuda::currentDevice();
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(error_d.get(), sizeof(GPU::SimpleVector<PixelErrorCompact>), device, stream));
  cudaCheck(cudaMemPrefetchAsync(data_d.get(), maxFedWords * sizeof(PixelErrorCompact), device, stream));
#endif
}

void SiPixelDigiErrorsCUDA::prefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(error_d.get(), sizeof(GPU::SimpleVector<PixelErrorCompact>), device, stream));
  cudaCheck(cudaMemPrefetchAsync(data_d.get(), maxFedWords_ * sizeof(PixelErrorCompact), device, stream));
#endif
}
