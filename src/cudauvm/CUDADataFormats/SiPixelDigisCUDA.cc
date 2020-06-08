#include "CUDADataFormats/SiPixelDigisCUDA.h"

#ifndef CUDAUVM_MANAGED_TEMPORARY
#include "CUDACore/device_unique_ptr.h"
#endif
#include "CUDACore/copyAsync.h"
#include "CUDACore/ScopedSetDevice.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream) {
#ifdef CUDAUVM_MANAGED_TEMPORARY
  xx_d = cms::cuda::make_managed_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::cuda::make_managed_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::cuda::make_managed_unique<uint16_t[]>(maxFedWords, stream);
#else
  xx_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
#endif
  adc_d = cms::cuda::make_managed_unique<uint16_t[]>(maxFedWords, stream);
  clus_d = cms::cuda::make_managed_unique<int32_t[]>(maxFedWords, stream);

  pdigi_d = cms::cuda::make_managed_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d = cms::cuda::make_managed_unique<uint32_t[]>(maxFedWords, stream);

  view_d = cms::cuda::make_managed_unique<DeviceConstView>(stream);
  view_d->xx_ = xx_d.get();
  view_d->yy_ = yy_d.get();
  view_d->adc_ = adc_d.get();
  view_d->moduleInd_ = moduleInd_d.get();
  view_d->clus_ = clus_d.get();

  device_ = cms::cuda::currentDevice();
#ifndef CUDAUVM_DISABLE_ADVISE
  cudaCheck(cudaMemAdvise(view_d.get(), sizeof(DeviceConstView), cudaMemAdviseSetReadMostly, device_));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(view_d.get(), sizeof(DeviceConstView), device_, stream));
#endif
}

SiPixelDigisCUDA::~SiPixelDigisCUDA() {
#ifndef CUDAUVM_DISABLE_ADVISE
  if (view_d) {
    // need to make sure a CUDA context is initialized for a thread
    cms::cuda::ScopedSetDevice(0);
    cudaCheck(cudaMemAdvise(view_d.get(), sizeof(DeviceConstView), cudaMemAdviseUnsetReadMostly, device_));
  }
#endif
}

void SiPixelDigisCUDA::adcPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(adc_d.get(), nDigis(), device, stream));
#endif
}

void SiPixelDigisCUDA::clusPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(clus_d.get(), nDigis(), device, stream));
#endif
}

void SiPixelDigisCUDA::pdigiPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(pdigi_d.get(), nDigis(), device, stream));
#endif
}

void SiPixelDigisCUDA::rawIdArrPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(rawIdArr_d.get(), nDigis(), device, stream));
#endif
}
