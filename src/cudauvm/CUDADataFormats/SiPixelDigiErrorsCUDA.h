#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include "DataFormats/PixelErrors.h"
#include "CUDACore/managed_unique_ptr.h"
#include "CUDACore/GPUSimpleVector.h"

#include <cuda_runtime.h>

class SiPixelDigiErrorsCUDA {
public:
  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cudaStream_t stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  GPU::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
  GPU::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
  GPU::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

  void prefetchAsync(int device, cudaStream_t stream) const;

private:
  cms::cuda::managed::unique_ptr<PixelErrorCompact[]> data_d;
  cms::cuda::managed::unique_ptr<GPU::SimpleVector<PixelErrorCompact>> error_d;
  PixelFormatterErrors formatterErrors_h;
  size_t maxFedWords_;
};

#endif
