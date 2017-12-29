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
"""Tests for Softplus and SoftplusGrad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SoftplusTest(test.TestCase):

  def _npSoftplus(self, np_features):
    np_features = np.asarray(np_features)
    zero = np.asarray(0).astype(np_features.dtype)
    return np.logaddexp(zero, np_features)

  def _testSoftplus(self, np_features, use_gpu=False):
    np_softplus = self._npSoftplus(np_features)
    with self.test_session(use_gpu=use_gpu):
      softplus = nn_ops.acl_softplus(np_features)
      tf_softplus = softplus.eval()
    self.assertAllCloseAccordingToType(np_softplus, tf_softplus)
    self.assertTrue(np.all(tf_softplus > 0))
    self.assertShapeEqual(np_softplus, softplus)

  def _testAclSoftplus(self, np_features, use_gpu=False):
    np_softplus = self._npSoftplus(np_features)
    with self.test_session(use_gpu=use_gpu):
      acl_softplus = nn_ops.acl_softplus(np_features)
      tf_softplus = acl_softplus.eval()
    self.assertAllCloseAccordingToType(np_softplus, tf_softplus)
    self.assertTrue(np.all(tf_softplus > 0))
    self.assertShapeEqual(np_softplus, acl_softplus)
 
  def testAclNumbers(self):
    for t in [np.float32]:
      self._testAclSoftplus(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      log_eps = np.log(np.finfo(t).eps)
      one = t(1)
      ten = t(10)
      self._testAclSoftplus(
          [
              log_eps,
              #log_eps - one,
              log_eps + one,
              #log_eps - ten,
              log_eps + ten, -log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=False)
"""
  def testWarnInts(self):
    # Running the op triggers address sanitizer errors, so we just make it
    nn_ops.acl_softplus(constant_op.constant(7))
"""

if __name__ == "__main__":
  test.main()
