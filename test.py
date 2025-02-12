import onnx

model = onnx.load("last.onnx")
print([node.op_type for node in model.graph.node])
