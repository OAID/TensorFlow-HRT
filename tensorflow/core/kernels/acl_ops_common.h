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
#ifndef TENSORFLOW_CORE_KERNELS_ACL_OPS_HPP_
#define TENSORFLOW_CORE_KERNELS_ACL_OPS_HPP_

#if defined(USE_ACL)

#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NELocallyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLLocallyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLBatchNormalizationLayer.h"
#include "arm_compute/core/NEON/kernels/NEDepthConcatenateKernel.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConcatenate.h"
#include "arm_compute/core/CL/kernels/CLDepthConcatenateKernel.h"
#include "arm_compute/runtime/CL/functions/CLDepthConcatenate.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace tensorflow {

enum TensorType{
    tensor_input,
    tensor_output,
    tensor_weights,
    tensor_biases,
    tensor_data,
};

template <typename ACLTensor>
class BaseTensor:public ACLTensor{
public:
    BaseTensor(bool share)
       :share_(share),type_(tensor_input),allocate_(false){
    }
    virtual void bindmem(void *mem,bool share){
        mem_=mem;
        share_=share;
    }
    virtual void settensortype(TensorType type){
        type_=type;
    };
    virtual void map(bool blocking = true){}
    virtual void unmap(){}
    virtual void commit(TensorType type=tensor_data);
    int tensor_copy(void * mem, bool toTensor=true);
protected:
    void* mem_;
    bool share_;
    TensorType type_;
    bool allocate_;
};

class GPUTensor:public BaseTensor<arm_compute::CLTensor>{
public:
    explicit GPUTensor(bool share)
       :BaseTensor(share){}
    virtual void map(bool blocking = true){
        if (!allocate_){
            arm_compute::CLTensor::allocator()->allocate();
            allocate_=true;
        }
        arm_compute::CLTensor::map(blocking);
     }
     virtual void unmap(){
        arm_compute::CLTensor::unmap();
     }
};

class CPUTensor:public BaseTensor<arm_compute::Tensor>{
public:
    explicit CPUTensor(bool share)
        :BaseTensor(share){}
    virtual void map(bool blocking = true){
        if (!allocate_){
            arm_compute::Tensor::allocator()->allocate();
            allocate_=true;
        }
    }
    virtual void unmap(){
    }
};

template <typename ACLLayer,typename ACLTensor>
class ACLXPUBaseLayer{
public:
    virtual void commit(){
        if (input) {
            input->commit(tensor_input);
        }
        if (output){
            output->commit(tensor_output);
        }
        if (weights){
            weights->commit(tensor_weights);
        }
        if (biases){
            biases->commit(tensor_biases);
        }
    }
    
    virtual void run(bool gpu){
        commit();
        layer->run();
        if (gpu) {
            // Make sure all the OpenCL jobs are done executing:
            arm_compute::CLScheduler::get().sync();
        }
    }
    
    virtual bool reshape(arm_compute::TensorShape &shape,TensorType type);
    explicit ACLXPUBaseLayer(){
        layer=nullptr;
        input=nullptr;
        output=nullptr;
        weights=nullptr;
        biases=nullptr;
        mean=nullptr;
        var=nullptr;
        beta=nullptr;
        gamma=nullptr;
    }
    
    virtual void freelayer(){
        if (layer) delete layer;
        if (input) delete input;
        if (output) delete output;
        if (weights) delete weights;
        if (biases) delete biases;
        if (mean) delete mean;
        if (var) delete var;
        if (beta) delete beta;
        if (gamma) delete gamma;
        layer=nullptr;
        input=nullptr;
        output=nullptr;
        weights=nullptr;
        biases=nullptr;
        mean=nullptr;
        var=nullptr;
        beta=nullptr; 
        gamma=nullptr;
    }
 
    virtual ~ACLXPUBaseLayer(){
        freelayer();
    }
    ACLLayer *layer;
    ACLTensor *input;
    ACLTensor *output;
    ACLTensor *weights;
    ACLTensor *biases;
    //for BN
    ACLTensor *mean;
    ACLTensor *var;
    ACLTensor *beta; 
    ACLTensor *gamma;
};

template <typename GPULayer, typename CPULayer>
class ACLBaseLayer {
public:
    explicit ACLBaseLayer();
    virtual void gpu_run();
    virtual void cpu_run();
    virtual ~ACLBaseLayer();
    virtual GPULayer * new_gpulayer();
    virtual CPULayer * new_cpulayer();
    ACLXPUBaseLayer<GPULayer,GPUTensor>& gpu(){
        return gpu_;
    }
    ACLXPUBaseLayer<CPULayer,CPUTensor>& cpu(){
        return cpu_;
    }
    bool checkreshape(arm_compute::TensorShape shape,bool gpu=false, TensorType type=tensor_input);
    void acl_run(void *input_data, void *output_data,bool gpu=false);
    template <typename ACLTensor> bool tensor_mem(ACLTensor *tensor,void *mem,bool share=false);
    template <typename ACLTensor> bool tensor_mem(void *mem,ACLTensor *tensor,bool share=false);
    template <typename ACLTensor> bool new_tensor(ACLTensor *&tensor,arm_compute::TensorShape shape,
						  void *mem=nullptr,bool share=false);
protected:
    ACLXPUBaseLayer<GPULayer,GPUTensor> gpu_;
    ACLXPUBaseLayer<CPULayer,CPUTensor> cpu_;
    bool init_layer_;
    bool force_bypass_acl_path_;
};

}  // namespace tensorflow

#define acl_configure(xlayer, input, args...) \
            xlayer.layer->configure(input, args);

#define INSTANTIATE_ACLBASECLASS(GPULayer,CPULayer) \
  template class ACLBaseLayer<GPULayer,CPULayer>; 

#define INSTANTIATE_ACLBASE_FUNCTION(GPULayer,CPULayer,ACLTensor) \
    template bool ACLBaseLayer<GPULayer,CPULayer>::tensor_mem(ACLTensor *tensor,void *mem,bool share); \
    template bool ACLBaseLayer<GPULayer,CPULayer>::tensor_mem(void *mem,ACLTensor *tensor,bool share); \
    template bool ACLBaseLayer<GPULayer,CPULayer>::new_tensor(ACLTensor *&tensor,arm_compute::TensorShape shape,void *mem,bool share); \

#define is_gpu_  (std::is_same<Device, Eigen::GpuDevice>::value)


#if defined(TEST_ACL)

#define NHWC2NCHW(in, out, N, C, H, W)        \
  do {                                        \
    for (int n = 0; n < N; n++)               \
      for (int c = 0; c < C; c++)             \
        for (int h = 0; h < H; h++)           \
          for (int w = 0; w < W; w++)         \
            out[n*C*H*W + c*H*W + h*W + w] =  \
              in[n*H*W*C + h*W*C + w*C + c];  \
  } while(0)

#define NCHW2NHWC(in, out, N, C, H, W)        \
  do {                                        \
    for (int n = 0; n < N; n++)               \
      for (int c = 0; c < C; c++)             \
        for (int h = 0; h < H; h++)           \
          for (int w = 0; w < W; w++)         \
            out[n*H*W*C + h*W*C + w*C + c] =  \
              in[n*C*H*W + c*H*W + h*W + w];  \
  } while(0)

#define HWIO2OIHW(in, out, H, W, I, O)        \
  do {                                        \
    for (int h = 0; h < H; h++)               \
      for (int w = 0; w < W; w++)             \
        for (int i = 0; i < I; i++)           \
          for (int o = 0; o < O; o++)         \
            out[o*I*H*W + i*W*H + h*W + w] =  \
              in[h*W*I*O + w*I*O + i*O + o];  \
  } while(0)
#endif

#endif

#endif

