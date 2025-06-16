##############################
# cuda_convolution_padded.jl
# GPU-accelerated RGBA convolution with explicit padding using CUDA.jl
# Uses raw UInt8 arrays of shape (H,W,4) instead of ColorTypes.RGBA
##############################

using CUDA                    
using FileIO                  
using ImageIO                 
using ColorTypes
using FixedPointNumbers

# Convolution parameters
const KERNEL_SIZE = 5
const PADDING     = div(KERNEL_SIZE, 2)
const COEFF       = Int32(273)

# 5×5 Gaussian-like kernel (host & device)
const h_kernel = Int32[
    1  4  7  4 1;
    4 16 26 16 4;
    7 26 41 26 7;
    4 16 26 16 4;
    1  4  7  4 1
]
const d_kernel = CuArray(h_kernel)

"""
pad_kernel!
  GPU kernel: zero-pad 3D UInt8 src into 3D UInt8 dst,
  where dimensions are (H+2PADDING)×(W+2PADDING)×4 channels.
"""
function pad_kernel!(dst::CuDeviceArray{ColorTypes.RGB{FixedPointNumbers.N0f8},2},
                     src::CuDeviceArray{ColorTypes.RGB{FixedPointNumbers.N0f8},2},
                     H::Int, W::Int)
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    Hp, Wp= size(dst)
    if i <= Hp && j <= Wp
        y = i - PADDING
        x = j - PADDING
        if 1 <= y <= H && 1 <= x <= W
            @inbounds for c in 1:C
                dst[i,j,c] = src[y,x,c]
            end
        else
            @inbounds for c in 1:C
                dst[i,j,c] = 0
            end
        end
    end
    return
end

"""
conv_kernel!
  GPU kernel: Each thread computes one output pixel from padded UInt8 src.
"""
function conv_kernel!(out::CuDeviceArray{ColorTypes.RGB{FixedPointNumbers.N0f8}, 2},
                      padded::CuDeviceArray{ColorTypes.RGB{FixedPointNumbers.N0f8}, 2},
                      kernel::CuDeviceArray{Int32,2},
                      H::Int, W::Int)
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= H && j <= W
        # accumulators
        acc = ntuple(_->Int32(0), 4)
        @inbounds for ky in 1:KERNEL_SIZE, kx in 1:KERNEL_SIZE
            k = kernel[ky, kx]
            for c in 1:4
                acc_c = acc[c] + Int32(padded[i+ky-1, j+kx-1, c]) * k
                acc = Base.setindex(acc, acc_c, c)
            end
        end
        @inbounds for c in 1:4
            val = acc[c] ÷ COEFF
            out[i,j,c] = UInt8(clamp(val, 0, 255))
        end
    end
    return
end

function main()
    # Load BMP: yields Array{Colorant,2}; convert to UInt8 H×W×4
    img = load("tester.bmp")
    H, W = size(img, 1), size(img, 2)

    # Move to GPU
    d_src = CuArray(img)
    # Allocate padded and output
    Hp, Wp   = H+2*PADDING, W+2*PADDING
    d_padded = CUDA.zeros(ColorTypes.RGB{FixedPointNumbers.N0f8}, Hp, Wp )
    d_out    = CUDA.zeros(ColorTypes.RGB{FixedPointNumbers.N0f8}, H, W)

    # Launch config
    threads     = (16,16)
    pad_blocks  = (cld(Wp, threads[1]), cld(Hp, threads[2]))
    conv_blocks = (cld(W,  threads[1]), cld(H,  threads[2]))

    # Pad then convolve on GPU
    @cuda threads=threads blocks=pad_blocks pad_kernel!(d_padded, d_src, H, W)
    synchronize()
    @cuda threads=threads blocks=conv_blocks conv_kernel!(d_out, d_padded, d_kernel, H, W)
    synchronize()

    # Retrieve and save
    out = Array(d_out)                 # H×W×4 UInt8
    # back to 4×H×W for saving
    out_chw = permutedims(out, (3,1,2))
    img_out = colorview(RGBA, out_chw)
    save("output.bmp", img_out)
    println("Wrote output.bmp with explicit padding and CUDA.jl")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
