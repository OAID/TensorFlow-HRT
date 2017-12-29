/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/nn_ops.cc.
#if defined(USE_ACL)

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/acl_conv_ops.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


#define REGISTER_ACL_CONV_OP(D, N, T)                       \
REGISTER_KERNEL_BUILDER(Name(N)                             \
                        .Device(DEVICE_##D)                 \
                        .TypeConstraint<T>("T"),            \
                        AclConv2DOp<D##Device, T,           \
                          arm_compute::CLConvolutionLayer,  \
                          arm_compute::NEConvolutionLayer>);

REGISTER_ACL_CONV_OP(CPU, "AclConv2D", float);
REGISTER_ACL_CONV_OP(GPU, "AclConv2D", float);

}  // namespace tensorflow
#endif //USE_ACL