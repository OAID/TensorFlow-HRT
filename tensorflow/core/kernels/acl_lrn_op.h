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

// LRN = Local Response Normalization
// See docs in ../ops/nn_ops.cc.
#if defined(USE_ACL)

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/kernels/acl_ops_common.h"

namespace tensorflow {

template <typename Device, typename T>
class AclLRNOp : public OpKernel,
  public ACLBaseLayer<arm_compute::CLNormalizationLayer, 
    arm_compute::NENormalizationLayer> {
 public:
  explicit AclLRNOp(OpKernelConstruction* context) : OpKernel(context),
    type_(arm_compute::NormType::IN_MAP_1D) {
 
    int64 depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(context, FastBoundsCheck(depth_radius64,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("depth_radius = ", depth_radius64,
                                        " larger than int max"));
    depth_radius_ = static_cast<int>(depth_radius64);
    float tmp;
    OP_REQUIRES_OK(context, context->GetAttr("bias", &tmp));
    bias_ = T(tmp);
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &tmp));
    alpha_ = T(tmp);
    OP_REQUIRES_OK(context, context->GetAttr("beta", &tmp));
    beta_ = T(tmp);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& in = context->input(0);
    OP_REQUIRES(context, in.dims() == 4,
                errors::InvalidArgument("in must be 4-dimensional"));
    OP_REQUIRES(context, FastBoundsCheck(in.NumElements(),
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("argument to LRN too large"));

    // Cast to platform-specific int to avoid conversion warnings.
    const int dim[4] = {
      static_cast<int>(in.dim_size(0)),
      static_cast<int>(in.dim_size(1)),
      static_cast<int>(in.dim_size(2)),
      static_cast<int>(in.dim_size(3))
    };
  
    OP_REQUIRES(context,
                (dim[1] + depth_radius_) <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("depth ", dim[1], " + depth_radius ",
                                        depth_radius_, " exceeds int max."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0,
                   TensorShape({dim[0], dim[1], dim[2], dim[3]}), &output));

    RunACLLayer(context, in, output, dim);
  }

 private:
  void RunACLLayer(OpKernelContext* ctx, const Tensor& in_data,
                   Tensor* out_data, const int* dim) {
    arm_compute::TensorShape shape(dim[3], dim[2], dim[1]);
    checkreshape(shape,is_gpu_);
    if (!this->init_layer_) return;
    if (is_gpu_) new_gpulayer();
    else new_cpulayer();

    this->force_bypass_acl_path_ = false;
    arm_compute::NormalizationLayerInfo* norm_info;
    const T* input_data  = in_data.flat<T>().data();
    T* output_data = out_data->flat<T>().data();;

    const float nsize = depth_radius_ + depth_radius_ + 1;
    const float scale = (bias_ == 1) ? alpha_ * nsize : alpha_;

    norm_info = new arm_compute::NormalizationLayerInfo(type_, nsize,
                                                        scale, beta_, bias_);

    if (is_gpu_) {
      new_tensor(this->gpu().input, shape, (void*)input_data);
      new_tensor(this->gpu().output, shape, (void*)output_data);
      acl_configure(this->gpu(), this->gpu().input, this->gpu().output, *norm_info);
    }else{
      new_tensor<CPUTensor>(this->cpu().input, shape, (void*)input_data);
      new_tensor<CPUTensor>(this->cpu().output, shape, (void*)output_data);
      acl_configure(this->cpu(), this->cpu().input, this->cpu().output, *norm_info);
    }
    delete norm_info;

    for (unsigned int n = 0; n < dim[0]; ++n) {
      acl_run((void*)input_data, (void*)output_data, is_gpu_);
      input_data  += dim[3] * dim[2] * dim[1];
      output_data += dim[3] * dim[2] * dim[1];
    }
  }

 private:
  int depth_radius_;
  T bias_;
  T alpha_;
  T beta_;
  const arm_compute::NormType type_;
};

}  // namespace tensorflow
#endif