#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#pragma clang diagnostic pop

using uchar4 = sycl::uchar4;
using uint4 = sycl::uint4;

constexpr int kernel_size = 5;
constexpr int padding = kernel_size/2;
constexpr int coeff = 273;
constexpr int kernel[kernel_size * kernel_size] = 
      {1,  4,  7,  4, 1,
       4, 16, 26, 16, 4,
       7, 26, 41, 26, 7,
       4, 16, 26, 16, 4,
       1,  4,  7,  4, 1};

int main()
{
    int width, height, channels;
    uchar4* host_data;

    host_data = reinterpret_cast<uchar4*>(
        stbi_load("obraz.bmp", &width, &height, &channels, 4));

    const auto pitchWithPadding = width + 2 * padding;
    const auto sizeWithPadding = pitchWithPadding * (height + 2 * padding);
    const auto sizeWithoutPadding = width * height;
    const auto skipTopRows = pitchWithPadding * 2 + 2;

    sycl::queue q{};

    uchar4* paddedData = sycl::malloc_device<uchar4>(sizeWithPadding, q);
    uchar4* targetData = sycl::malloc_device<uchar4>(sizeWithPadding, q);

    q.fill(paddedData, uchar4{0, 0, 0, 0}, sizeWithPadding);
    q.ext_oneapi_copy2d(host_data, width, paddedData + skipTopRows, pitchWithPadding, width, height);

    int* kernel_device = sycl::malloc_device<int>(kernel_size * kernel_size, q);
    q.copy<int>(&kernel[0], kernel_device, kernel_size * kernel_size);

    auto kernel_range = sycl::range<2>(width, height);

    q.submit([&](auto& h){
        h.parallel_for(kernel_range, [=](auto idx){

            int x = idx[0] + padding;
            int y = idx[1] + padding;
            int coord =  y * pitchWithPadding + x;
            uint4 sum{};

            for(int ky = -padding; ky <= padding; ++ky) {
                for(int kx = -padding; kx <= padding; ++kx) {
                
                    int ki = (ky + padding) * kernel_size
                           + (kx + padding);

                    int  off = ky * pitchWithPadding + kx;
                    auto pix = paddedData[coord + off].convert<uint>();
    
                    sum += kernel_device[ki] * pix;
                }
            }

            sum /= coeff;

            targetData[coord] = sum.convert<uint8_t>();
        });
    });

    q.ext_oneapi_copy2d(targetData + skipTopRows, pitchWithPadding, host_data, width, width, height);

    stbi_write_bmp("output.bmp", width, height, 4, host_data);

    return 0;
}
