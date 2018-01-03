/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_ACL_POOLING_OPS_H_
#define TENSORFLOW_CORE_KERNELS_ACL_POOLING_OPS_H_

#if defined(USE_ACL)

#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/avgpooling_op.h"
#include "tensorflow/core/kernels/maxpooling_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/kernels/acl_ops_common.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

// A helper class to manage sizes and shapes for pooling operations.
struct AclPoolParameters {
  // Updates context->status if there is an invalid input.
  AclPoolParameters(OpKernelContext* context, const std::vector<int32>& ksize,
                 const std::vector<int32>& stride, Padding padding,
                 TensorFormat data_format, const TensorShape& tensor_in_shape);

  int depth; //dim

  int tensor_in_cols;  //width_
  int tensor_in_rows;  //height_
  int tensor_in_batch; //num_

  int window_rows;  //kernel_h_ 
  int window_cols;  //kernel_w_
  int depth_window;  //channels_

  int row_stride;	 //stride_h_
  int col_stride;	 //stride_w_
  int depth_stride; 

  int64 out_height; //pooled_height_
  int64 out_width;	 //pooled_width_
  int out_depth;

  int64 pad_rows; //pad_h_
  int64 pad_cols; //pad_w_
  int pad_depth;
 
  TensorFormat data_format;
  
  // Returns the shape of the output for "forward" pooling operations.
  TensorShape forward_output_shape();  
};

template <typename Device, typename T>
class AclPoolingOp : public OpKernel,
  public ACLBaseLayer<arm_compute::CLPoolingLayer,arm_compute::NEPoolingLayer>  {
 public:
  explicit AclPoolingOp(OpKernelConstruction* context, bool pool_type) : OpKernel(context), is_max_pool_(pool_type) {
    string data_format;
    auto status = context->GetAttr("data_format", &data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else
      data_format_ = FORMAT_NCHW;
#if !defined(TEST_ACL)
    OP_REQUIRES(
        context, data_format_ == FORMAT_NCHW,
        errors::Unimplemented("Only supports FORMAT_NCHW."));
#endif

    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    round_type_ = static_cast<arm_compute::DimensionRoundingType>(
      static_cast<int>(arm_compute::DimensionRoundingType::CEIL) + 1);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    AclPoolParameters params{context,  ksize_,      stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) return;

    OP_REQUIRES(context, AclCheckParamsInternal(params),
                errors::InvalidArgument("tensorflow data size is not same as ACL!")); 

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));

#if defined(TEST_ACL)
    if (data_format_ == FORMAT_NHWC) {
      Tensor in_tmp;
      Tensor out_tmp;

      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            tensor_in.shape(), &in_tmp));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            output->shape(), &out_tmp));
      NHWC2NCHW(tensor_in.flat<T>().data(), in_tmp.flat<T>().data(),
                   params.tensor_in_batch, params.depth,
                   params.tensor_in_rows, params.tensor_in_cols);
      RunAclLayer(context, params, in_tmp, &out_tmp);
      NCHW2NHWC(out_tmp.flat<T>().data(), output->flat<T>().data(),
                   params.tensor_in_batch, params.out_depth,
                   params.out_height, params.out_width);
      return;
    }
#endif
    RunAclLayer(context, params, tensor_in, output);
  }
  
  inline bool IsMaxpool() const { return  is_max_pool_; }

  bool AclCheckParams(OpKernelContext* context) {
    const Tensor& tensor_in = context->input(0);
    AclPoolParameters params{context,  ksize_,      stride_,
                          padding_, data_format_, tensor_in.shape()};
    return AclCheckParamsInternal(params);
  }

 private:
  bool AclCheckParamsInternal(const AclPoolParameters& params) {

    if (round_type_ == arm_compute::DimensionRoundingType::CEIL
        || round_type_ == arm_compute::DimensionRoundingType::FLOOR)
    return true;

    int pooled_w = -1, pooled_h= -1;
    std::tie(pooled_w, pooled_h) = arm_compute::scaled_dimensions(
      params.tensor_in_cols, params.tensor_in_rows,
      params.window_cols, params.window_rows,
      arm_compute::PadStrideInfo(params.col_stride, params.row_stride,
      params.pad_cols, params.pad_rows,
      arm_compute::DimensionRoundingType::CEIL));
    if (pooled_w == params.out_width && pooled_h == params.out_height) {
      round_type_ = arm_compute::DimensionRoundingType::CEIL;
      return true;
    }

    std::tie(pooled_w, pooled_h) = arm_compute::scaled_dimensions(
      params.tensor_in_cols, params.tensor_in_rows,
      params.window_cols, params.window_rows,
      arm_compute::PadStrideInfo(params.col_stride, params.row_stride,
      params.pad_cols, params.pad_rows,
      arm_compute::DimensionRoundingType::FLOOR));

    if (pooled_w == params.out_width && pooled_h == params.out_height) {
      round_type_ = arm_compute::DimensionRoundingType::FLOOR;
      return true;
    }
    return false;
  }

   void RunAclLayer(OpKernelContext* context, const AclPoolParameters& params,
                    const Tensor& tensor_in, Tensor* output) {

    const T* input_data = tensor_in.flat<T>().data();
    T* output_data = output->flat<T>().data();

    if (this->init_layer_) {
      arm_compute::TensorShape in_shape((unsigned int)params.tensor_in_cols,
          (unsigned int)params.tensor_in_rows,
          (unsigned int)params.depth);
      arm_compute::TensorShape out_shape((unsigned int)params.out_width,
            (unsigned int)params.out_height,
            (unsigned int)params.out_depth);
      checkreshape(in_shape,is_gpu_);

      this->init_layer_=false;
      if (is_gpu_) new_gpulayer();
      else new_cpulayer();

      this->force_bypass_acl_path_ = false;
      arm_compute::PoolingLayerInfo *pool_info;

      arm_compute::PoolingType pool_type = is_max_pool_ ? 
                  arm_compute::PoolingType::MAX :
                  arm_compute::PoolingType::AVG;

      pool_info = new arm_compute::PoolingLayerInfo(pool_type,
                    params.window_cols,
                    arm_compute::PadStrideInfo(params.col_stride,
                      params.row_stride,
                      params.pad_cols,
                      params.pad_rows,
                      round_type_));
      if (is_gpu_) {
        new_tensor(this->gpu().input, in_shape, (void*)input_data);
        new_tensor(this->gpu().output, out_shape, (void*)output_data);
        this->gpu().layer->configure(this->gpu().input,this->gpu().output,*pool_info);
      }else{
        new_tensor(this->cpu().input,in_shape,(void*)input_data);
        new_tensor(this->cpu().output,out_shape,(void*)output_data);
        this->cpu().layer->configure(this->cpu().input,this->cpu().output,*pool_info);
      }
      delete pool_info;
    }

    for (unsigned int n = 0; n < params.tensor_in_batch; ++n) {
      acl_run((void*)input_data, (void*)output_data, is_gpu_);
      input_data += params.tensor_in_rows * params.tensor_in_cols * params.depth;
      output_data += params.out_height * params.out_width * params.out_depth;
    }
  }

  // Single-threaded implementation of DepthwiseMaxPool which
  // does not handle all of the same options as SpatialMaxPool
  // (strict assumptions on no padding, stride).
  //
  // TODO(vrv): implement a more general depthwise-max pool that works
  // on GPU as well.

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  arm_compute::DimensionRoundingType round_type_;
  const bool is_max_pool_;
};

template <typename Device, typename T>
class AclMaxPoolingOp : public AclPoolingOp<Device, T> {
 public:
  explicit AclMaxPoolingOp(OpKernelConstruction* context) :
      AclPoolingOp<Device, T> (context, true){
  }
};

template <typename Device, typename T>
class AclAvgPoolingOp : public AclPoolingOp<Device, T> {
 public:
  explicit AclAvgPoolingOp(OpKernelConstruction* context) :
      AclPoolingOp<Device, T> (context, false) {
  }
};

} // namespace tensorflow

#endif  // USE_ACL
#endif  // TENSORFLOW_CORE_KERNELS_ACL_POOLING_OPS_H_
