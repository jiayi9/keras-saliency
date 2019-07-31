#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:20:00 2019

@author: Lu Jiayi
"""

# source:
# https://www.cnblogs.com/qiandeheng/p/10175188.html


from keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework import graph_io
from keras import backend as K

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
    return frozen_graph

K.set_learning_phase(0)
keras_model = load_model('/Users/tef-itm/Downloads/96_5cm.h5')
print('Inputs are:', keras_model.inputs)
print('Outputs are:', keras_model.outputs)

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
graph_io.write_graph(frozen_graph, "/Users/tef-itm/Documents/H5_CPP/", "96_5cm_freeze.pb", as_text=False)
