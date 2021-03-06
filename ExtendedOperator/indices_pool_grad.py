import tensorflow as tf
from tensorflow.python.framework import ops
import os

grad_module=tf.load_op_library(os.path.split(os.path.realpath(__file__))[0]+"/build/libIndicesPoolGrad.so")

@ops.RegisterGradient("IndicesPool")
def _indices_pool_grad(op,grad):
    return [grad_module.indices_pool_grad(
        op.inputs[0],op.inputs[1],grad,
        patch_num=op.get_attr('patch_num')),None]
