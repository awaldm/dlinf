#include "dlinf/layers/avgpool2d.hpp"

#include <Eigen/Core>
#include <stdexcept>

namespace dlinf {
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> avgpool2d_direct(
    const Eigen::Ref<const RowMatrixXf>& input,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding)
    {

        // input.rows() is number of channels (both input and output)

        if (input.cols() != static_cast<Eigen::Index>(height) * static_cast<Eigen::Index>(width)) {
            throw std::runtime_error("AvgPool2D flattened input size mismatch");
        }
        const Eigen::Index H_out = (height + 2 * padding - kernel_size) / stride + 1;
        const Eigen::Index W_out = (width + 2 * padding - kernel_size) / stride + 1;
    

    

    
        if (kernel_size <= 0) {
            throw std::runtime_error("AvgPool2D kernel dimensions must be positive");
        }


        if (H_out <= 0 || W_out <= 0) {
            throw std::runtime_error("AvgPool2D output shape is empty");
        }
        RowMatrixXf output(input.rows(), (H_out*W_out));

       
    

        for (Eigen::Index oc = 0; oc < input.rows(); ++oc) {
            for (Eigen::Index oy = 0; oy < H_out; ++oy) {
                for (Eigen::Index ox = 0; ox < W_out; ++ox) {
                    float sum = 0.0; // init output at that output tensor position
                    int num_valid = 0; // valid entries in the kernel
                        // Now we are at the output position, iterate through the kernel for this position
                        for (Eigen::Index ky = 0; ky < kernel_size; ++ky) { // y position inside kernel
                            for (Eigen::Index kx = 0; kx < kernel_size; ++kx) { // x position inside kernel
                                const Eigen::Index iy = oy * stride + ky - padding; // y index of ky relative to the output tensor
                                const Eigen::Index ix = ox * stride + kx - padding; // map kernel position to output coordinate
                                if (iy >= 0 && iy < height && ix >= 0 && ix < width) { // get only valid positions
                                                    
                                    sum += input(oc, iy * width + ix); //
                                    num_valid++; 
                                }
                            }
                        }
                    
                    output(oc, oy*W_out+ox) = sum / num_valid;
                }
            }
        }

    return output;
    }
}  // namespace dlinf