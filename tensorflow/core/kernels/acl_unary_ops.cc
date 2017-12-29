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

#if defined(USE_ACL)

#include "tensorflow/core/kernels/acl_unary_ops.h"
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_ACL_UNARY_OP1(D, N, F, T)                               \
  REGISTER_KERNEL_BUILDER( Name(N)                                       \
                           .Device(DEVICE_##D)                           \
                           .TypeConstraint<T>("T"),                      \
                           AclUnaryOp<D##Device, F<T>, T>);

#define REGISTER_ACL_UNARY_OP2(D, N, F, T)                               \
  REGISTER_KERNEL_BUILDER( Name(N)                                       \
                           .Device(DEVICE_##D)                           \
                           .TypeConstraint<T>("T"),                      \
                           AclUnaryOp<D##Device, F<D##Device, T>, T>);

REGISTER_ACL_UNARY_OP1(CPU, "AclSigmoid", functor::sigmoid, float);
REGISTER_ACL_UNARY_OP1(GPU, "AclSigmoid", functor::sigmoid, float);
REGISTER_ACL_UNARY_OP1(CPU, "AclTanh", functor::tanh, float);
REGISTER_ACL_UNARY_OP1(GPU, "AclTanh", functor::tanh, float);
REGISTER_ACL_UNARY_OP2(CPU, "AclRelu", functor::Relu, float);
REGISTER_ACL_UNARY_OP2(GPU, "AclRelu", functor::Relu, float);
REGISTER_ACL_UNARY_OP2(CPU, "AclSoftplus", functor::Softplus, float);
REGISTER_ACL_UNARY_OP2(GPU, "AclSoftplus", functor::Softplus, float);

#if defined(TEST_ACL) && 0
REGISTER_ACL_UNARY_OP1(CPU, "Sigmoid", functor::sigmoid, float);
REGISTER_ACL_UNARY_OP1(GPU, "Sigmoid", functor::sigmoid, float);
REGISTER_ACL_UNARY_OP1(CPU, "Tanh", functor::tanh, float);
REGISTER_ACL_UNARY_OP1(GPU, "Tanh", functor::tanh, float);
REGISTER_ACL_UNARY_OP2(CPU, "Relu", functor::Relu, float);
REGISTER_ACL_UNARY_OP2(GPU, "Relu", functor::Relu, float);
REGISTER_ACL_UNARY_OP2(CPU, "Softplus", functor::Softplus, float);
REGISTER_ACL_UNARY_OP2(GPU, "Softplus", functor::Softplus, float);
#endif
} // end namespace tensorflow
#endif // define USE_ACL
