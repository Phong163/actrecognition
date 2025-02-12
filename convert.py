# Import thư viện
import torch
from models.model_lstm import LSTMModel

# Load lại mô hình đã train
input_size = 32  # Số feature trên mỗi timestep
model = LSTMModel(input_size)
model.load_state_dict(torch.load(r"C:\Users\OS\Desktop\ActionProject\model_lstm_multiclass.pth"))
model.eval()  # Đưa về chế độ inference
dummy_input = torch.randn(1, 6, input_size)
torch.onnx.export(
    model,                      # Mô hình PyTorch
    dummy_input,                # Input mẫu
    "model_lstm_multiclass.onnx", # Tên file ONNX đầu ra
    export_params=True,         # Xuất cả tham số đã train
    opset_version=11,           # Chọn phiên bản ONNX (nên dùng 11 hoặc 12)
    input_names=["input"],      # Tên input
    output_names=["output"],    # Tên output
    dynamic_axes=None
)

print("Model converted to ONNX and saved as model_lstm_multiclass.onnx")
