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

// See docs in ../ops/math_ops.cc.
#if defined(USE_ACL)

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/matmul_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/util/matmul_autotune.h"
#include "tensorflow/core/kernels/acl_ops_common.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AclMatMulOp : public OpKernel,
  public ACLBaseLayer<arm_compute::CLFullyConnectedLayer,arm_compute::NEFullyConnectedLayer> {
 public:
  explicit AclMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES(ctx, transpose_a_ == 0,
                   errors::Unimplemented("Acl only does not support A transpose!"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));

  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    OP_REQUIRES(
        ctx, a.NumElements() && b.NumElements(),
        errors::InvalidArgument("Acl does not support empty matrix!"));

    RunACLLayer(ctx, a, b, out);
  }

 private:
  void RunACLLayer(OpKernelContext* ctx,  const Tensor& a,
                     const Tensor& b, Tensor* out) {

    unsigned int M = a.dim_size(0); 
    unsigned int N = transpose_b_ ? b.dim_size(0) : b.dim_size(1);
    unsigned int K = a.dim_size(1);

    OP_REQUIRES(
      ctx, M > 1 && N > 1 && K > 1,
      errors::InvalidArgument("Acl does not support 1D multiply!"));

    arm_compute::TensorShape input_shape(K, M);
    arm_compute::TensorShape output_shape(N, M);
    checkreshape(input_shape,is_gpu_);
    if (!this->init_layer_) return;
    this->init_layer_=false;
    if (is_gpu_) new_gpulayer();
    else new_cpulayer();

    const T* input_data = a.flat<T>().data();
    T* output_data = out->flat<T>().data();
    const T* weithts_data= b.flat<T>().data();

    this->force_bypass_acl_path_ = false; 
    if (is_gpu_) {
        if (transpose_b_) {
            new_tensor(this->gpu().weights, arm_compute::TensorShape(K, N), (void*)weithts_data);
        }else{
            new_tensor(this->gpu().weights, arm_compute::TensorShape(N, K), (void*)weithts_data);
        }
        tensor_mem(this->gpu().weights, (void*)weithts_data);
        new_tensor(this->gpu().input, input_shape, (void*)input_data);
        new_tensor(this->gpu().output, output_shape, (void*)output_data);
        acl_configure(this->gpu(), this->gpu().input, this->gpu().weights,
                      this->gpu().biases, this->gpu().output, transpose_b_);
    }else{
        if (transpose_b_) {
            new_tensor(this->cpu().weights, arm_compute::TensorShape(K, N), (void*)weithts_data);
        }else{
            new_tensor(this->cpu().weights, arm_compute::TensorShape(N, K), (void*)weithts_data);
        }
        tensor_mem(this->cpu().weights, (void*)weithts_data);
        new_tensor(this->cpu().input, input_shape, (void*)input_data);
        new_tensor(this->cpu().output, output_shape, (void*)output_data);
        acl_configure(this->cpu(), this->cpu().input, this->cpu().weights,
                      this->cpu().biases, this->cpu().output, transpose_b_);
    }

    acl_run((void*)input_data, (void*)output_data, is_gpu_);
  }

 private:
  std::vector<int64> algorithms_;
  bool algorithms_set_already_;
  bool use_autotune_;
  bool transpose_a_;
  bool transpose_b_;
};

}  // namespace tensorflow
#endif // USE_ACL