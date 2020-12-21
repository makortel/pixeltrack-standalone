#include <iostream>

#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  std::cout << "World" << std::endl;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<KokkosExecSpace>(0, 4),
      KOKKOS_LAMBDA(const size_t i) { printf("Kokkos::parallel_for loop element %lu\n", i); });
  return 0;
}
