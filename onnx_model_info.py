import onnx
import sys

from onnx import ModelProto

model_file = sys.argv[1]

model = ModelProto()

print("-- Opening ONNX file=%s" % model_file)

f = open(model_file, "rb")
model.ParseFromString(f.read())
f.close()

print("-- ONNX OpSet=%s" % model.opset_import)

print("-- ONNX model - Number of nodes=%d" % len(model.graph.node))

print("-- ONNX model dump=\n%s" % model)

onnx.checker.check_model(model)
print('-- ONNX model validated OK')
