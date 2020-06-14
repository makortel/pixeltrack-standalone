#include "CUDADataFormats/SiPixelClustersCUDA.h"

#ifndef CUDAUVM_MANAGED_TEMPORARY
#include "CUDACore/host_unique_ptr.h"
#endif
#include "CUDACore/copyAsync.h"
#include "CUDACore/ScopedSetDevice.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxClusters, cudaStream_t stream) {
  moduleStart_d = cms::cuda::make_managed_unique<uint32_t[]>(maxClusters + 1, stream);
  clusModuleStart_d = cms::cuda::make_managed_unique<uint32_t[]>(maxClusters + 1, stream);
#ifdef CUDAUVM_MANAGED_TEMPORARY
  clusInModule_d = cms::cuda::make_managed_unique<uint32_t[]>(maxClusters, stream);
  moduleId_d = cms::cuda::make_managed_unique<uint32_t[]>(maxClusters, stream);
#else
  clusInModule_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
  moduleId_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
#endif

  view_d = cms::cuda::make_managed_unique<DeviceConstView>(stream);
  view_d->moduleStart_ = moduleStart_d.get();
  view_d->clusInModule_ = clusInModule_d.get();
  view_d->moduleId_ = moduleId_d.get();
  view_d->clusModuleStart_ = clusModuleStart_d.get();

  device_ = cms::cuda::currentDevice();
#ifndef CUDAUVM_DISABLE_ADVISE
  cudaCheck(cudaMemAdvise(view_d.get(), sizeof(DeviceConstView), cudaMemAdviseSetReadMostly, device_));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(view_d.get(), sizeof(DeviceConstView), device_, stream));
#endif
}

SiPixelClustersCUDA::~SiPixelClustersCUDA() {
#ifndef CUDAUVM_DISABLE_ADVISE
  if (view_d) {
    // need to make sure a CUDA context is initialized for a thread
    cms::cuda::ScopedSetDevice(0);
    cudaCheck(cudaMemAdvise(view_d.get(), sizeof(DeviceConstView), cudaMemAdviseUnsetReadMostly, device_));
  }
#endif
}
