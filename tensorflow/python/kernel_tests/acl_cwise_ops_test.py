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
"""Functional tests for coefficient-wise operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
_POW = lambda x, y: x**y
_TRUEDIV = lambda x, y: x / y
_FLOORDIV = lambda x, y: x // y
_MOD = lambda x, y: x % y
_NEG = lambda x: -x
_ABS = abs

_LT = lambda x, y: x < y
_LE = lambda x, y: x <= y
_GT = lambda x, y: x > y
_GE = lambda x, y: x >= y

_AND = lambda x, y: x & y
_OR = lambda x, y: x | y
_XOR = lambda x, y: x ^ y
_INV = lambda x: ~x


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), x_values

def _default_tolerance(dtype):
  """Returns a sensible default tolerance for comparing results of a given
  type"""
  if dtype == np.float16:
    return 5e-3
  elif dtype in (np.float32, np.complex64):
    return 1e-3
  elif dtype in (np.float64, np.complex128):
    return 1e-5
  else:
    return None # Fail fast for unexpected types


class UnaryOpTest(test.TestCase):

  def _compareCpu(self, x, np_func, tf_func, grad_rtol=None, grad_atol=None):
    if grad_rtol is None:
      grad_rtol = _default_tolerance(x.dtype)
    if grad_atol is None:
      grad_atol = _default_tolerance(x.dtype)
    np_ans = np_func(x)
    with self.test_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      if x.dtype in (np.float32, np.float64):
        y = 1.1 * tf_func(inx)
        np_ans *= 1.1
      else:
        y = tf_func(inx)
      tf_cpu = y.eval()

      self.assertShapeEqual(np_ans, y)
      if x.dtype == np.float16:
        self.assertAllClose(np_ans, tf_cpu, rtol=1e-3, atol=1e-3)
      else:
        self.assertAllClose(np_ans, tf_cpu)

      return
      if x.dtype in (np.complex64, np.complex128) and tf_func == math_ops.sign:
        return  # Return early

      if x.dtype == np.float16:
        s = list(np.shape(x))
        jacob_t, _ = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x)
        xf = x.astype(np.float)
        inxf = ops.convert_to_tensor(xf)
        yf = tf_func(inxf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, yf, s, x_init_value=xf, delta=1e-2)
        jacob_n = jacob_n.astype(np.float16)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
      elif x.dtype in (np.float32, np.complex64):
        s = list(np.shape(x))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x, delta=1e-3)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
      elif x.dtype in (np.float64, np.complex128):
        s = list(np.shape(x))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x, delta=1e-5)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)

  def _check(self, result_tensor, result_np, input_sp_t, tol):
    self.assertTrue(isinstance(result_tensor, sparse_tensor.SparseTensor))
    self.assertTrue(isinstance(input_sp_t, sparse_tensor.SparseTensor))
    self.assertAllEqual(input_sp_t.indices.eval(), result_tensor.indices.eval())
    self.assertAllEqual(input_sp_t.dense_shape.eval(),
                        result_tensor.dense_shape.eval())
    if tol is None:
      self.assertAllClose(result_np, result_tensor.values.eval())
    else:
      self.assertAllClose(
          result_np, result_tensor.values.eval(), rtol=tol, atol=tol)

  def _compareSparseCpu(self, x, np_func, tf_func, tol):
    x_sp, x_sp_vals = _sparsify(x)
    res_np = np_func(x_sp_vals)
    with self.test_session(use_gpu=False):
      self._check(tf_func(x_sp), res_np, x_sp, tol)

  def _compareGpu(self, x, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=True):
      result = tf_func(ops.convert_to_tensor(x))
      tf_gpu = result.eval()
    if x.dtype == np.float16:
      self.assertAllClose(np_ans, tf_gpu, rtol=1e-3, atol=1e-3)
    else:
      self.assertAllClose(np_ans, tf_gpu)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareSparseGpu(self, x, np_func, tf_func, tol):
    x_sp, x_sp_vals = _sparsify(x)
    res_np = np_func(x_sp_vals)
    with self.test_session(use_gpu=True):
      self._check(tf_func(x_sp), res_np, x_sp, tol)

  def _compareBoth(self, x, np_func, tf_func):
    self._compareCpu(x, np_func, tf_func)
    self._compareGpu(x, np_func, tf_func)

  def _sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def _tanh(self, x):
    return np.tanh(x)

  def _replace_domain_error_with_inf(self, fn):

    def func(x):
      try:
        return fn(x)
      except ValueError as e:
        if "domain error" in str(e):
          return np.inf * np.ones_like(x)
        else:
          raise e

    return func

  def testFloatBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    w = x - x.min() + 1.02  # all greater than 1
    y = (x + .5).astype(np.float32)  # no zero
    z = (x + 15.5).astype(np.float32)  # all positive
    k = np.arange(-0.90, 0.90, 0.25).astype(np.float32)  # between -1 and 1

    self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareBoth(x, self._sigmoid, math_ops.acl_sigmoid)
    self._compareBoth(x, self._tanh, math_ops.tanh)
    self._compareBoth(x, self._tanh, math_ops.acl_tanh)
if __name__ == "__main__":
  test.main()
