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
"""Tests for Relu and ReluGrad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def _elu_grad_grad(activation):
  if activation < 0:
    return np.exp(activation)
  return 0


class ReluTest(test.TestCase):

  def _npRelu(self, np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7, 0.9]
                     ])))

  def _testRelu(self, np_features, use_gpu=False):
    np_relu = self._npRelu(np_features)
    with self.test_session(use_gpu=use_gpu):
      relu = nn_ops.relu(np_features)
      tf_relu = relu.eval()
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, relu)

  def _testAclRelu(self, np_features, use_gpu=False):
    np_relu = self._npRelu(np_features)
    with self.test_session(use_gpu=use_gpu):
      relu = nn_ops.acl_relu(np_features)
      tf_relu = relu.eval()
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, relu)


  def testNumbers(self):
    for t in [np.float32]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      self._testAclRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)

if __name__ == "__main__":
  test.main()
