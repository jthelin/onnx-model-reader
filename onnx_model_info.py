# Copyright 2019 Jorgen Thelin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import onnx.utils
import sys

model_file = sys.argv[1]

print("-- Opening ONNX file=%s" % model_file)
model = onnx.load_model(model_file) # type: onnx.ModelProto

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
all_passes = onnx.optimizer.get_available_passes()
print("-- Available optimization passes:")
for p in all_passes:
    print(p)
print()

# Run model checker, optimizer, & shape inference engine on the model, and also strip any doc_string's.
print("-- Optimizing and polishing ONNX model")
polished_model = onnx.utils.polish_model(model)

print("-- ONNX polished model - Number of nodes=%d" % len(polished_model.graph.node))
