#pragma once

#include <vector>
#include "containers/CSR.hpp"

namespace algorithms {

    std::vector<float> deltaStepping(sycl::queue& q, containers::CSR& csr, int source, int delta);

    void test(sycl::queue&q);
} // namespace algorithms