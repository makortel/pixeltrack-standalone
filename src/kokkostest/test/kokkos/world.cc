#include <iostream>

#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  std::cout << "World" << std::endl;

  Kokkos::View<float *, KokkosExecSpace> a("a", 4);
  auto h_a = Kokkos::create_mirror_view(a);
  for (int i = 0; i < 4; i++) {
    h_a[i] = i;
  }
  Kokkos::deep_copy(a, h_a);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<KokkosExecSpace>(0, 4),
      KOKKOS_LAMBDA(const size_t i) {
        printf("Kokkos::parallel_for loop element %lu\n", i);
        a.data()[0] = 0;
      });
  return 0;
}
