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

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/softmax_op_functor.h"
#include "tensorflow/core/kernels/acl_ops_common.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AclSoftmaxOp : public OpKernel,
  public ACLBaseLayer<arm_compute::CLSoftmaxLayer,arm_compute::NESoftmaxLayer> {

 public:
  explicit AclSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, logits_in.shape(), &softmax_out));
    if (logits_in.NumElements() > 0)
      RunACLLayer(context,logits_in, softmax_out); 
  }

 private:
  void RunACLLayer(OpKernelContext* context, const Tensor& softmax_in,
                   Tensor* softmax_out){

    const T* input_data = softmax_in.flat<T>().data();
    T* output_data = softmax_out->flat<T>().data();
    const uint64 batch_size = softmax_in.dim_size(0);
    const uint64 num_classes = softmax_in.dim_size(1);

    if (this->init_layer_) {
      arm_compute::TensorShape shape(num_classes);
      checkreshape(shape, is_gpu_);
      this->init_layer_= false;

      if (is_gpu_) new_gpulayer();
      else new_cpulayer();

      this->force_bypass_acl_path_ = false;

      if (is_gpu_) {
          new_tensor(this->gpu().input, shape, (void*)input_data);
          new_tensor(this->gpu().output, shape, (void*)output_data);
          acl_configure(this->gpu(), this->gpu().input, this->gpu().output);
      }else{
          new_tensor(this->cpu().input, shape, (void*)input_data);
          new_tensor(this->cpu().output, shape, (void*)output_data);
          acl_configure(this->cpu(), this->cpu().input, this->cpu().output);
      }
    }

    for (unsigned int i = 0; i < batch_size; ++i) {
        acl_run((void*)input_data, (void*)output_data, is_gpu_);
        output_data += num_classes;
        input_data += num_classes;
    }
  }

};

}  // namespace tensorflow

#endif //USE_ACL