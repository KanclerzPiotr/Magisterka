#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct uchar4 {
    uint8_t r, g, b, a;
};

constexpr int kernel_size = 5;
constexpr int padding = kernel_size/2;
constexpr int coeff = 273;
constexpr int kernel[kernel_size * kernel_size] = 
      {1,  4,  7,  4, 1,
       4, 16, 26, 16, 4,
       7, 26, 41, 26, 7,
       4, 16, 26, 16, 4,
       1,  4,  7,  4, 1};

std::ostream& operator<<(std::ostream& os, uchar4 const& v) {
    // print as { r, g, b, a }
    os << '['
       << std::setw(3) << static_cast<int>(v.r) << " "
       << std::setw(3) << static_cast<int>(v.g) << " "
       << std::setw(3) << static_cast<int>(v.b) << " "
       << std::setw(3) << static_cast<int>(v.a)
       << ']';
    return os;
}

int main()
{
    int width, height, channels;
    uchar4* host_data;

    host_data = reinterpret_cast<uchar4*>(
        stbi_load("tester.bmp", &width, &height, &channels, 4));

    const auto widthWithPadding = width + 2 * padding;
    const auto heightWithPadding = height + 2 * padding;
    const auto sizeWithPadding = widthWithPadding * heightWithPadding;
    const auto size = width * height;
    

    uchar4* paddedData = new uchar4[sizeWithPadding];
    uchar4* targetData = new uchar4[size];

    #pragma omp target data \
        map(host_data[0:size], kernel[0:kernel_size * kernel_size]) \
        map(width, height, widthWithPadding, heightWithPadding, padding) \
        map(paddedData[0:sizeWithPadding]) \
        map(targetData[0:size])
    {
        #pragma omp target teams distribute parallel for
        for(int i = 0; i < heightWithPadding; i++) {
            for(int j = 0; j < widthWithPadding; j++) {

                
                if(i >= padding && i < heightWithPadding - padding && j >= padding && j < widthWithPadding - padding ) {
                    uchar4 pix = host_data[ (i- padding) * width + (j - padding)];
                    paddedData[i * widthWithPadding + j] = pix;
                }
                else
                    paddedData[i * widthWithPadding + j] = uchar4{0,0,0,0};
                
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                uint32_t r{0}, g{0}, b{0}, a{0};
                int paddedIdx = (i+padding) * widthWithPadding + (j + padding);

                for(int ky = -padding; ky <= padding; ++ky) {
                    for(int kx = -padding; kx <= padding; ++kx) {
                        int ki = (ky + padding) * kernel_size
                               + (kx + padding);
                        int off = ky * widthWithPadding + kx;
                        uchar4 pix = paddedData[paddedIdx + off];
                        r += pix.r * kernel[ki];
                        g += pix.g * kernel[ki];
                        b += pix.b * kernel[ki];
                        a += pix.a * kernel[ki];
                    }
                }
                uchar4 res = {r/coeff, g/coeff, b/coeff, a/coeff};
                targetData[i* width + j] = res;
            }
        }
        
    }

    stbi_write_bmp("output.bmp", width, height, 4, targetData);

    return 0;
}