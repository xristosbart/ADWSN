# Filename: to_onnx.py
# Authors: Christos Gklezos, Ahwar
# Source: https://stackoverflow.com/questions/53182177/how-do-you-convert-a-onnx-to-tflite

import torch
import numpy as np
model = torch.load("cluster_ae.pt")
model.eval()

x = torch.randn(1, 6, requires_grad=True)
torch_out = model(x)

torch.onnx.export(model,                     # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "cluster_ae.onnx",         # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names

import onnx

onnx_model = onnx.load("cluster_ae.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("cluster_ae.onnx")
#
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

import tensorflow as tf
from onnx_tf.backend import prepare

tf_rep = prepare(onnx_model)
tf_rep.export_graph("cluster.tf")


converter = tf.lite.TFLiteConverter.from_saved_model("cluster.tf")
tflite_model = converter.convert()

# Save the model
with open("cluster.tflite", 'wb') as f:
    f.write(tflite_model)


tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)



interpreter = tf.lite.Interpreter(model_path="cluster.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
input_shape = input_details[0]['shape']
input_data = to_numpy(x)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# get_tensor() returns a copy of the tensor data
# use tensor() in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(torch_out)
print(output_data)