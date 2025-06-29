module Blur

using CUDA
using Images

const KERNEL_SIZE = 5
const PADDING     = div(KERNEL_SIZE, 2)
const COEFF       = Int32(273)

const h_kernel = Int32[
    1  4  7  4 1;
    4 16 26 16 4;
    7 26 41 26 7;
    4 16 26 16 4;
    1  4  7  4 1
]
const d_kernel = CuArray(h_kernel)

function copy_pad_image_kernel_rgba(src, dst, padding)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    h, w = size(dst)

    if i > padding && i <= h - padding &&
       j > padding && j <= w - padding
        dst[i, j] = src[i - padding, j - padding]
    end
    return
end

function convolve(
    src :: CuDeviceMatrix{RGBA{N0f8}},
    out :: CuDeviceMatrix{RGBA{N0f8}},
    kernel :: CuDeviceMatrix{Int32})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    h, w = size(out)
    
    if i <= h && j <= w
        ip = i + PADDING
        jp = j + PADDING

        r = Int32(0)
        g = Int32(0)
        b = Int32(0)
        a = Int32(0)

        @inbounds for ky = 1:KERNEL_SIZE
            row = ip + ky - PADDING - 1
            @inbounds for kx  = 1:KERNEL_SIZE
                col = jp + kx - PADDING - 1
                k = kernel[ky, kx]
                pix = src[row, col]
                r += k * reinterpret(UInt8, pix.r)
                g += k * reinterpret(UInt8, pix.g)
                b += k * reinterpret(UInt8, pix.b)
            end
        end

        out_r = UInt8(div(r, COEFF))
        out_g = UInt8(div(g, COEFF))
        out_b = UInt8(div(b, COEFF))
        out_a = UInt8(255)

        out[i, j] = RGBA{N0f8}(
            reinterpret(N0f8, out_r),
            reinterpret(N0f8, out_g),
            reinterpret(N0f8, out_b),
            reinterpret(N0f8, out_a))
    end
    return
end

function main()
    img = load("obraz.bmp")
    H, W = size(img, 1), size(img, 2)
    Hp, Wp   = H+2*PADDING, W+2*PADDING

    rgba_img = similar(img, RGBA{N0f8}) 
    map!(RGBA, rgba_img, img) 
    d_src = CuArray(rgba_img)
    d_padded = CUDA.zeros(RGBA{N0f8}, Hp, Wp )

    threads     = (16,16)
    pad_blocks  = (cld(Wp, threads[1]), cld(Hp, threads[2]))
    conv_blocks = (cld(W,  threads[1]), cld(H,  threads[2]))

    @cuda threads=threads blocks=pad_blocks copy_pad_image_kernel_rgba(d_src, d_padded, PADDING)
    @cuda threads=threads blocks=conv_blocks convolve(d_padded, d_src, d_kernel)

    # Retrieve and save
    copyto!(rgba_img, d_src)         
    save("output.bmp", rgba_img)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end