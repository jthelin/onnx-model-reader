import onnx
import onnx.utils
import sys

from onnx import ModelProto
from onnx import optimizer

model_file = sys.argv[1]

print("-- Opening ONNX file=%s" % model_file)
model: ModelProto = onnx.load(model_file)

print("-- ONNX OpSet=%s" % model.opset_import)

print("-- ONNX model - Number of nodes=%d" % len(model.graph.node))

print()
print("-- Begin ONNX model --")
# Print a human readable representation of the model graph
print(onnx.helper.printable_graph(model.graph))
print("-- End ONNX model --")
print()

onnx.checker.check_model(model)
print('-- ONNX model validated OK')
print()

# A full list of supported optimization passes can be found using get_available_passes()
all_passes = optimizer.get_available_passes()
print("-- Available optimization passes:")
for p in all_passes:
    print(p)
print()

# Run model checker, optimizer, & shape inference engine on the model, and also strip any doc_string's.
print("-- Optimizing and polishing ONNX model")
polished_model = onnx.utils.polish_model(model)

print("-- ONNX polished model - Number of nodes=%d" % len(polished_model.graph.node))
