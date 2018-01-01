import tensorflow as tf
from tensorflow.python.framework import ops

import os

grad_module=tf.load_op_library(os.path.split(os.path.realpath(__file__))[0]+"/build/libGlobalIndicesPoolGrad.so")

@ops.RegisterGradient("GlobalIndicesPool")
def _global_indices_pool_grad(op,grad):
    return [grad_module.global_indices_pool_grad(
            op.inputs[0],op.inputs[1],grad),None]
