# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional tests for convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np

from tensorflow.contrib import layers
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def GetShrunkInceptionShapes(shrink=10):
  """Iterator for smaller versions of convolution shapes in 2015 Inception.

  Relative to inception, each depth value is `depth // shrink`.

  Args:
    shrink: Factor to shrink each depth value by relative to Inception.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the convolution
    parameters of Inception layers.
  """
  input_sizes = [[4, 5, 5, 1248], [4, 8, 8, 384], [4, 8, 8, 384],
                 [4, 8, 8, 2048], [4, 8, 8, 448], [4, 8, 8, 2048],
                 [4, 8, 8, 2048], [4, 8, 8, 2048], [4, 8, 8, 1760],
                 [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 8, 8, 1760],
                 [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1248],
                 [4, 17, 17, 128], [4, 17, 17, 1248], [4, 17, 17, 224],
                 [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1216],
                 [4, 17, 17, 1216], [4, 17, 17, 224], [4, 17, 17, 192],
                 [4, 17, 17, 192], [4, 17, 17, 1152], [4, 17, 17, 1152],
                 [4, 17, 17, 192], [4, 17, 17, 160], [4, 17, 17, 1152],
                 [4, 17, 17, 1024], [4, 17, 17, 128], [4, 17, 17, 1024],
                 [4, 17, 17, 128], [4, 17, 17, 1024], [4, 17, 17, 128],
                 [4, 17, 17, 768], [4, 17, 17, 128], [4, 17, 17, 128],
                 [4, 17, 17, 768], [4, 17, 17, 768], [4, 35, 35, 96],
                 [4, 35, 35, 288], [4, 35, 35, 64], [4, 35, 35, 288],
                 [4, 35, 35, 256], [4, 35, 35, 48], [4, 35, 35, 256],
                 [4, 35, 35, 96], [4, 35, 35, 192], [4, 35, 35, 192],
                 [4, 35, 35, 192], [4, 73, 73, 64], [4, 73, 73, 64],
                 [4, 147, 147, 24]]
  
  filter_sizes = [[1, 1, 1248, 128], [1, 1, 384, 384], [1, 1, 384, 384],
                  [1, 1, 2048, 192], [3, 3, 448, 384], [1, 1, 2048, 320],
                  [1, 1, 2048, 448], [1, 1, 2048, 384], [1, 1, 1760, 384],
                  [1, 1, 1760, 192], [1, 1, 1760, 448], [1, 1, 1760, 320],
                  [3, 3, 192, 192], [3, 3, 192, 192], [1, 1, 1248, 192],
                  [3, 3, 128, 320], [1, 1, 1248, 128], [1, 1, 224, 224],
                  [3, 3, 192, 256], [3, 3, 192, 256], [1, 1, 1216, 192],
                  [1, 1, 1216, 96], [1, 1, 224, 224], [3, 3, 192, 224],
                  [3, 3, 192, 192], [1, 1, 1152, 192], [1, 1, 1152, 128],
                  [3, 3, 192, 192], [3, 3, 160, 192], [1, 1, 1152, 160],
                  [1, 1, 1024, 128], [3, 3, 128, 192], [1, 1, 1024, 160],
                  [3, 3, 128, 192], [1, 1, 1024, 256], [3, 3, 128, 128],
                  [1, 1, 768, 192], [3, 3, 128, 128], [3, 3, 128, 128],
                  [1, 1, 768, 128], [1, 1, 768, 320], [3, 3, 96, 96],
                  [3, 3, 288, 384], [3, 3, 64, 96], [1, 1, 288, 64],
                  [1, 1, 256, 64], [3, 3, 48, 64], [1, 1, 256, 48],
                  [3, 3, 96, 96], [1, 1, 192, 32], [1, 1, 192, 64],
                  [1, 1, 192, 48], [3, 3, 64, 192], [1, 1, 64, 64],
                  [1, 1, 24, 64]]
  out_sizes = [[4, 5, 5, 128], [4, 8, 8, 384], [4, 8, 8, 384],
               [4, 8, 8, 192], [4, 8, 8, 384], [4, 8, 8, 320],
               [4, 8, 8, 448], [4, 8, 8, 384], [4, 8, 8, 384],
               [4, 8, 8, 192], [4, 8, 8, 448], [4, 8, 8, 320],
               [4, 8, 8, 192], [4, 17, 17, 192], [4, 17, 17, 192],
               [4, 8, 8, 320], [4, 17, 17, 128], [4, 17, 17, 224],
               [4, 17, 17, 256], [4, 17, 17, 256], [4, 17, 17, 192],
               [4, 17, 17, 96], [4, 17, 17, 224], [4, 17, 17, 224],
               [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 128],
               [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 160],
               [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 160],
               [4, 17, 17, 192], [4, 17, 17, 256], [4, 17, 17, 128],
               [4, 17, 17, 192], [4, 17, 17, 128], [4, 17, 17, 128],
               [4, 17, 17, 128], [4, 17, 17, 320], [4, 17, 17, 96],
               [4, 17, 17, 384], [4, 35, 35, 96], [4, 35, 35, 64],
               [4, 35, 35, 64], [4, 35, 35, 64], [4, 35, 35, 48],
               [4, 35, 35, 96], [4, 35, 35, 32], [4, 35, 35, 64],
               [4, 35, 35, 48], [4, 71, 71, 192], [4, 73, 73, 64],
               [4, 147, 147, 64]]
  strides = [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1
  ]

  # Shrink sizes to make the test faster
  for i in input_sizes:
    i[3] //= shrink
  for f in filter_sizes:
    f[2] //= shrink
    f[3] //= shrink
  for o in out_sizes:
    o[3] //= shrink
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [
      SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      VALID, SAME, SAME, VALID, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, SAME, VALID, VALID, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, VALID, VALID, VALID
  ]
  for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                           paddings):
    yield i, f, o, s, p

def HWIOToOIHW(input_tensor):
  """Convert the input from HWIO format to OIHW.

  Args:
    input_tensor:  a 4-D tensor, or a 4-element array representing the same.

  Returns:
    the converted tensor or a shape array
  """
  if isinstance(input_tensor, ops.Tensor):
    return array_ops.transpose(input_tensor, [3, 2, 0, 1])
  else:
    return [input_tensor[3], input_tensor[2], input_tensor[0], input_tensor[1]]


def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NHWC", False), ("NHWC", True)]
  if test.is_gpu_available(cuda_only=True):
    # "NCHW" format is only supported on CUDA.
    test_configs += [("NCHW", True)]
  return test_configs

class Conv2DTest(test.TestCase):

  def _DtypesToTest(self, use_gpu):
    if use_gpu and not test_util.CudaSupportsHalfMatMulAndConv():
      return [dtypes.float32]
    else:
      # It is important that float32 comes before float16 here,
      # as we will be using its gradients as reference for fp16 gradients.
      return [dtypes.float32]

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, strides,
                            padding, data_format, dtype, use_gpu):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_gpu: True if the operations should be run on GPU
    Returns:
      Symbolic tensor value that can be used to execute the computation
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
      conv = nn_ops.conv2d(
          t1, t2, strides=strides, padding=padding, data_format=data_format)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv

  def _CompareFwdValues(self, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that CPU and GPU produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    x2 = np.random.rand(*filter_in_sizes).astype(np.float32)

    def _SetupVal(data_format, use_gpu):
      with test_util.device(use_gpu):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t1 = test_util.NHWCToNCHW(t1)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d(
            t1, t2, strides=strides, padding=padding, data_format=data_format)
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        return conv

    tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      tensors.append(_SetupVal(data_format, use_gpu))
    values = self.evaluate(tensors)
    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-5, atol=1e-5)

  def _AclSetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, strides,
                            padding, data_format, dtype, use_gpu):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_gpu: True if the operations should be run on GPU
    Returns:
      Symbolic tensor value that can be used to execute the computation
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 + 1 for f in range(total_size_1)]
    x2 = [f * 1.0 + 1 for f in range(total_size_2)]
    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
        t2 = HWIOToOIHW(t2)
      conv = nn_ops.acl_conv2d(
          t1, t2, strides=strides, padding=padding, data_format=data_format)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv
 
  def _AclCompareFwdValues(self, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that CPU and GPU produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    x2 = np.random.rand(*filter_in_sizes).astype(np.float32)    

    def _SetupVal(data_format, filter_format, use_gpu):
      with test_util.device(use_gpu):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        strides = [1] + conv_strides + [1]
        if filter_format == "OIHW":
          t2 = HWIOToOIHW(t2)
        if data_format == "NCHW":
          t1 = test_util.NHWCToNCHW(t1)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.acl_conv2d(
            t1, t2, strides=strides, padding=padding, data_format=data_format, filter_format=filter_format)
        return conv

    tensors = []
    for (data_format, filter_format, use_gpu) in [("NCHW", "OIHW", False) ]:
      tensors.append(_SetupVal(data_format, filter_format, use_gpu))
    values = self.evaluate(tensors)
    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-5, atol=1e-5)


  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, strides, padding,
                    expected):
    tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      for dtype in self._DtypesToTest(use_gpu):
        result = self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            strides,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu)
        tensors.append(result)
      values = self.evaluate(tensors)
      for i in range(len(tensors)):
        conv = tensors[i]
        value = values[i]
        tol = 1e-5
        if value.dtype == np.float16:
          tol = 1e-3
        self.assertAllClose(expected, np.ravel(value), atol=tol, rtol=tol)
        self.assertShapeEqual(value, conv)

  def _AclVerifyValues(self, tensor_in_sizes, filter_in_sizes, strides, padding,
                    expected):
    tensors = []
    for (data_format, use_gpu) in [ ("NCHW", False) ]:
      for dtype in self._DtypesToTest(use_gpu):
        result = self._AclSetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            strides,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu)
        tensors.append(result)
      values = self.evaluate(tensors)

      for i in range(len(tensors)):
        conv = tensors[i]
        value = values[i]
        tol = 1e-5
        if value.dtype == np.float16:
          tol = 1e-3
        self.assertAllClose(expected, np.ravel(value), atol=tol, rtol=tol)
        self.assertShapeEqual(value, conv)

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D1x1Filter(self):
    expected_output = [
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

    self._AclVerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes()
  def testConv2DKernelSmallerThanStrideSame(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1],
        strides=[2, 2],
        padding="SAME",
        expected=[1, 3, 7, 9])

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[1, 1, 1, 1],
        strides=[2, 2],
        padding="SAME",
        expected=[1, 3, 9, 11])

class DeepConv2DTest(test.TestCase):

  def _CompareFwdConv2D(self, data_format, filter_format, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that DeepConv2D and Conv2D produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    #x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    #x2 = np.random.rand(*filter_in_sizes).astype(np.float32)
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.test_session(use_gpu=False) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      strides = [1] + conv_strides + [1]
      acl_strides = strides #test_util.NHWCToNCHW(strides)
      if data_format == "NCHW":
        acl_t1 = test_util.NHWCToNCHW(t1)
      else:
        acl_t1 = t1
      if filter_format == "OIHW":
        acl_t2 = HWIOToOIHW(t2)
      else:
        acl_t2 = t2

      conv = nn_ops.conv2d(t1, t2, strides=strides, padding=padding)
      acl_conv = nn_ops.acl_conv2d(acl_t1, acl_t2, strides=acl_strides, padding=padding,
                                   data_format=data_format, filter_format=filter_format)
      os.environ["TF_USE_DEEP_CONV2D"] = "0"
      values_expect = sess.run([conv])
      values_test = sess.run([acl_conv])
      self.assertAllClose(values_expect, values_test, rtol=1e-5, atol=1e-5)

  def _RunTestCases(self, conv_strides, padding):
    input_sizes =  [[1, 4, 4, 2]]
    filter_sizes = [[3, 3, 2, 2]]
    for data_format, filter_format, input_shape, filter_shape in zip(["NHWC", "NCHW"], ["HWIO", "OIHW"], input_sizes, filter_sizes):
      self._CompareFwdConv2D(data_format, filter_format, input_shape, filter_shape, conv_strides, padding)

  def testConv2D3x3FilterStride1x1Valid(self):
    self._RunTestCases([1, 1], "VALID")

  def testConv2D3x3FilterStride1x1Same(self):
    self._RunTestCases([1, 1], "SAME")


class Conv2DBenchmark(test.Benchmark):

  def benchmarkGPUConvStackFirst(self):
    # Benchmark the first iteration of a conv-net with many identical conv
    # operations.
    if not test.is_gpu_available():
      return

    with ops.Graph().as_default(), session_lib.Session() as session:
      batch_size = 1
      timesteps = 600
      features = 1

      inputs = random_ops.random_uniform(
          [batch_size, 1, timesteps, features], seed=1234)
      num_outputs_list = [512] * 40 + [1]
      kernel_w = 3
      x = inputs
      for num_outputs in num_outputs_list:
        x = layers.convolution2d(x, num_outputs, [1, kernel_w])
      outputs = x

      variables.global_variables_initializer().run()
      num_iterations = 4
      for iter_index in xrange(num_iterations):
        start = time.time()
        session.run(outputs)
        wall_time = time.time() - start
        self.report_benchmark(
            name="conv_stack_iter_%d" % iter_index, wall_time=wall_time)
        print("conv_stack_iter_%d: %.4f" % (iter_index, wall_time))

def GetInceptionFwdTest(input_size, filter_size, stride, padding,
                        gpu_only=False):

  def Test(self):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping InceptionFwd %s", (input_size, filter_size,
                                                   stride, padding))
      return
    print("Testing InceptionFwd %s", (input_size, filter_size, stride,
                                                padding))
    tf_logging.info("Testing InceptionFwd %s", (input_size, filter_size, stride,
                                                padding))
    self._AclCompareFwdValues(input_size, filter_size, [stride, stride], padding)

  return Test

if __name__ == "__main__":
  for index, (input_size_, filter_size_, output_size_, stride_,
              padding_) in enumerate(GetShrunkInceptionShapes()):
    print("testInceptionFwd_"+ str(index), input_size_, filter_size_, output_size_, stride_)
    setattr(Conv2DTest, "testInceptionFwd_" + str(index),
            test_util.run_in_graph_and_eager_modes()(
                GetInceptionFwdTest(input_size_, filter_size_, stride_,
                                    padding_)))

  # TODO(b/35359731)
  # Fwd, BckInput, and BackFilter to test that for certain input parameter
  # set, winograd nonfused algorithm will be excluded from conv autotune. If
  # in such case, winograd nonfused algorithm is added as one option of the
  # conv autotune, and cuDNN version is smaller than 7, the following tests
  # will fail.
  ishape = [1, 400, 400, 1]
  fshape = [1, 1, 1, 256]
  oshape = [1, 400, 400, 256]
  setattr(Conv2DTest, "testInceptionFwd_No_Winograd_Nonfused",
          test_util.run_in_graph_and_eager_modes()(
              GetInceptionFwdTest(ishape, fshape, 1, "SAME", gpu_only=True)))
  test.main()
