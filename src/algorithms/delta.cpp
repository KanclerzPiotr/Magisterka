#include "algorithms/delta.hpp"

#include <sycl/sycl.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include 

namespace algorithms {

    template <typename T>
    bool atomic_fetch_min(sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref, T val) {
        T old = ref.load();
        while (val < old) {
            if (ref.compare_exchange_weak(old, val)) {
                return true;
            }
        }
        return false;
    }

    // vertex, value, delta, tent, buckets
    void relax(int vtx, float val, int d, float* t, int* b)
    {
        if (val < t[vtx])
        {
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> ref_t(t[vtx]);
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> ref_b(b[vtx]);
            atomic_fetch_min(ref_t, val);    
            atomic_fetch_min(ref_b, (int)val/d);
        }
    }

    bool areAllBucketsEmpty(auto policy, int* buckets, int size) {
        return std::all_of(policy, buckets, buckets + size, [](int i) { return i == -1; });
    }

    std::vector<float> deltaStepping(sycl::queue& q, containers::CSR& csr, int source, int delta) {
// 
        auto policy = oneapi::dpl::execution::make_device_policy(q);
        auto size = csr.size;
        float* tent = sycl::malloc_shared<float>(size, q);
        int * buckets = sycl::malloc_shared<int>(size, q);
        int * tempBuckets = sycl::malloc_shared<int>(size, q);



        q.submit([&](sycl::handler& h) {
            h.parallel_for(size, [=](sycl::id<1> i) {
                tent[i] = std::numeric_limits<float>::max();
                buckets[i] = std::numeric_limits<int>::max();

                if(i == source) {
                    relax(i, 0, delta, tent, buckets);
                }
            });
        });

        while( !areAllBucketsEmpty(policy, buckets, size)) {

            int bucket = *std::min_element(policy, buckets, buckets + size);

            q.submit([&](sycl::handler& h) {
                h.parallel_for(size, [=](sycl::id<1> i) {
                        if(buckets[i] == bucket) {

                        for(int j = csr.row[i]; j < csr.row[i+1]; j++) {
                            if (csr.val[j] < delta) { // light edge
                                relax(csr.col[j], tent[i] + csr.val[j], delta, tent, buckets)
                            }
                            
                        }

                        buckets[i] =    
                        }
                    });
                });
            };
        
        sycl::free(tent, q);
        sycl::free(buckets, q);

        return std::vector<float>{};
    };

    void test(sycl::queue&q) {

    }

} // namespace algorithms






//get light and heavy edges
//reduce to get numbers
//allocate data for light and heavy edges
//save indices of light and heavy edges per vertex
// write edges to light and heavy arrays
