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

#include "tensorflow/core/kernels/conv_ops.h"
#include <string.h>
#include <map>
#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/kernels/acl_ops_common.h"

namespace tensorflow {

template <typename Device, typename T, typename GPUConvLayer, typename CPUConvLayer>
class AclConv2DOp : public OpKernel,
    public ACLBaseLayer<GPUConvLayer, CPUConvLayer> {

  struct AclConv2DArgs {
    // Input layer dimensions
    int batch;
    int in_rows;
    int in_cols;
    int in_depth;
    int filter_rows;
    int filter_cols;
    int stride_rows;
    int stride_cols;
    int64 pad_rows;
    int64 pad_cols;

    // Output layer dimensions
    int64 out_rows;
    int64 out_cols;
    int out_depth;
  };

 public:
  explicit AclConv2DOp(OpKernelConstruction* context) :  OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format_str, filter_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
#if !defined(TEST_ACL)
    OP_REQUIRES(context, data_format_str == "NCHW",
                errors::Unimplemented("Acl only support NCHW format: ",
                                      data_format_str));
#endif
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format",
                data_format_str));

    if (!context->GetAttr("filter_format", &filter_format_str).ok())
      filter_format_str = "HWIO";

    OP_REQUIRES(context, FilterFormatFromString(filter_format_str, &filter_format_),
                errors::InvalidArgument("Invalid filter format string: ",
                                    filter_format_str));
#if !defined(TEST_ACL)
    OP_REQUIRES_OK(context, context->GetAttr("no_bias", &no_bias_));
#else
    no_bias_ = true;
#endif
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    AclConv2DArgs args;
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    const Tensor* bias = NULL;
    if (!no_bias_) {
      bias = &context->input(2);
      OP_REQUIRES(context, bias->dims() == 1,
                  errors::InvalidArgument("bias must be 1-dimensional: ",
                                          bias->shape().DebugString()));
    }

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(
          context,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth - OIHW.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, in_depth == GetFilterDim(filter, filter_format_, 'I'),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter.dim_size(1)));
    OP_REQUIRES(
        context,
        FastBoundsCheck(in_depth, std::numeric_limits<int>::max()),
        errors::InvalidArgument("in_depth too large"));
    args.in_depth = static_cast<int>(in_depth);

    // The last dimension for filter is out_depth.
    args.out_depth = GetFilterDim(filter, filter_format_, 'O');

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    args.in_rows = static_cast<int>(input_rows_raw);
    args.filter_rows = GetFilterDim(filter, filter_format_, 'H');

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    args.in_cols = static_cast<int>(input_cols_raw);
    args.filter_cols = GetFilterDim(filter, filter_format_, 'W');

    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    args.batch = static_cast<int>(batch_raw);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    args.stride_rows = GetTensorDim(strides_, data_format_, 'H');
    args.stride_cols = GetTensorDim(strides_, data_format_, 'W');

    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(args.in_rows, args.filter_rows, args.stride_rows,
                                         padding_, &args.out_rows, &args.pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(args.in_cols, args.filter_cols, args.stride_cols,
                                         padding_, &args.out_cols, &args.pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, args.batch, args.out_rows, args.out_cols, args.out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    OP_REQUIRES(
        context,
        ((args.filter_cols ==  1 && args.filter_rows == 1 &&
          args.pad_cols == 0 && args.pad_rows == 0 ) ||
         (args.filter_cols ==  3 && args.filter_rows == 3 &&
          args.pad_cols <= 1 && args.pad_rows <= 1 )),
        errors::InvalidArgument("ACL only support 1x1(pad 0) and 3x3(pad <= 1)"));
#if defined(TEST_ACL)
    if (data_format_ == FORMAT_NHWC || filter_format_ == FORMAT_HWIO) {
      Tensor in_tmp, *in = const_cast<Tensor* >(&input);
      Tensor out_tmp, *out = output;
      Tensor filter_tmp, *kernel = const_cast<Tensor* >(&filter);

      if (data_format_ == FORMAT_NHWC) {
        OP_REQUIRES_OK(context,
                      context->allocate_temp(DataTypeToEnum<T>::v(),
                                              input.shape(), &in_tmp));
        OP_REQUIRES_OK(context,
                      context->allocate_temp(DataTypeToEnum<T>::v(),
                                              output->shape(), &out_tmp));
        NHWC2NCHW(input.flat<T>().data(), in_tmp.flat<T>().data(),
                  args.batch, args.in_depth,
                  args.in_rows, args.in_cols);
        in = &in_tmp;
        out = &out_tmp;
      }

      if (filter_format_ == FORMAT_HWIO) {
        OP_REQUIRES_OK(context,
                      context->allocate_temp(DataTypeToEnum<T>::v(),
                                              filter.shape(), &filter_tmp));
        HWIO2OIHW(filter.flat<T>().data(), filter_tmp.flat<T>().data(),
                  args.filter_rows, args.filter_cols,
                  args.in_depth, args.out_depth);
        kernel = &filter_tmp;
      }
      RunACLLayer(context, *in, *kernel, bias, out, args);

      if (data_format_ == FORMAT_NHWC)
        NCHW2NHWC(out_tmp.flat<T>().data(), output->flat<T>().data(),
            args.batch, args.out_depth,
            args.out_rows, args.out_cols);
      return;
    }
#endif
    RunACLLayer(context, input, filter, bias, output, args);

  }

 private:
    void RunACLLayer(OpKernelContext* ctx, const Tensor& in_data, const Tensor& filter,
                     const Tensor* bias, Tensor* out_data, const AclConv2DArgs& args){

      arm_compute::TensorShape input_shape(args.in_cols, args.in_rows,
                                           args.in_depth, args.batch); //wxhxchxnum
      ACLBaseLayer<GPUConvLayer, CPUConvLayer>::checkreshape(input_shape, is_gpu_);
      if (!this->init_layer_) return;
      this->init_layer_=false;
    // Initialize ACL.
      if (is_gpu_) {
          ACLBaseLayer<GPUConvLayer, CPUConvLayer>::new_gpulayer();
      }else{
          ACLBaseLayer<GPUConvLayer, CPUConvLayer>::new_cpulayer();
      }
      this->force_bypass_acl_path_=false;

      arm_compute::PadStrideInfo conv_info(args.stride_cols, args.stride_rows,
                                           args.pad_cols, args.pad_rows/*, round_type*/);
      arm_compute::TensorShape weights_shape(args.filter_cols, args.filter_rows,
                                             args.in_depth, args.out_depth);
      arm_compute::TensorShape biases_shape (args.out_depth);
      arm_compute::TensorShape output_shape(args.out_cols, args.out_rows,
                                            args.out_depth, args.batch);
      const T* input_data = in_data.flat<T>().data();
      T* output_data = out_data->flat<T>().data();
      T* weithts_data = const_cast<T* >(filter.flat<T>().data());
      const T* bias_data=nullptr;
      if (!no_bias_) 
          bias_data = bias->flat<T>().data();

      if (is_gpu_) {
          //[kernel_x, kernel_y, IFM, OFM]
          ACLBaseLayer<GPUConvLayer, CPUConvLayer>::new_tensor(
            this->gpu().weights, weights_shape, (void*)weithts_data);
          ACLBaseLayer<GPUConvLayer, CPUConvLayer>::tensor_mem(
            this->gpu().weights, (void*)weithts_data);
          //[OFM]
          if (!no_bias_) {
              ACLBaseLayer<GPUConvLayer,CPUConvLayer>::new_tensor(
                this->gpu().biases,biases_shape, (void*)bias_data);
              ACLBaseLayer<GPUConvLayer,CPUConvLayer>::tensor_mem(
                this->gpu().biases, (void*)bias_data);
          }

          //[width, height, IFM]
          ACLBaseLayer<GPUConvLayer,CPUConvLayer>::new_tensor(
            this->gpu().input, input_shape, (void*)input_data);
          //[width, height, OFM]
          ACLBaseLayer<GPUConvLayer,CPUConvLayer>::new_tensor(
            this->gpu().output, output_shape, (void*)output_data);
          acl_configure(this->gpu(), this->gpu().input, this->gpu().weights,
                        this->gpu().biases, this->gpu().output, conv_info);
      }else{
          //[kernel_x, kernel_y, IFM, OFM]
          ACLBaseLayer<GPUConvLayer,CPUConvLayer>::new_tensor(
            this->cpu().weights, weights_shape, (void*)weithts_data);
          ACLBaseLayer<GPUConvLayer,CPUConvLayer>::tensor_mem(
            this->cpu().weights, (void*)weithts_data);
          //[OFM]
          if (!no_bias_) {
              ACLBaseLayer<GPUConvLayer,CPUConvLayer>::new_tensor(
                this->cpu().biases, biases_shape, (void*)bias_data);
              ACLBaseLayer<GPUConvLayer,CPUConvLayer>::tensor_mem(
                this->cpu().biases, (void*)bias_data);
          }

          //[width, height, IFM]
          ACLBaseLayer<GPUConvLayer,CPUConvLayer>::new_tensor(
            this->cpu().input,
            input_shape, (void*)input_data);
          //[width, height, OFM]
          ACLBaseLayer<GPUConvLayer,CPUConvLayer>::new_tensor(
            this->cpu().output,
            output_shape, (void*)output_data);
          acl_configure(this->cpu(), this->cpu().input, this->cpu().weights,
                        this->cpu().biases, this->cpu().output, conv_info);
      }
      ACLBaseLayer<GPUConvLayer,CPUConvLayer>::acl_run((void*)input_data,(void*)output_data, is_gpu_);
  }
  
  std::vector<int32> strides_;
  bool no_bias_;
  Padding padding_;
  TensorFormat data_format_;
  FilterTensorFormat filter_format_;
  TF_DISALLOW_COPY_AND_ASSIGN(AclConv2DOp);
};

}  // namespace tensorflow
#endif //USE_ACL