import tensorflow as tf
from tensorflow.python.framework import ops

import os

grad_module=tf.load_op_library(os.path.split(os.path.realpath(__file__))[0]+"/build/libContextBlockPoolGrad.so")

@ops.RegisterGradient("ContextBlockPool")
def _context_block_pool_grad(op,grad):
    return [grad_module.context_block_pool_grad(
        op.inputs[0],op.inputs[1],op.inputs[2],grad),None,None]
