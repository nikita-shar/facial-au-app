import torch
import onnx
from custom_gcn import EdgeCNNMT
#from CorePyDEConv.ONNX_Tests.custom_gcn import EdgeCNNMT


class PosContainer:
    def __init__(self, pos):
        self.pos = pos

class ONNXWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        d = PosContainer(x)
        raw_output = self.base(d)
        return self.softmax(raw_output)


def load_model(checkpoint_path="checkpoints/checkpoint_96.tar"):
    model = EdgeCNNMT(k=20, aggr='max', num_aus=24, num_classes=3)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    #print("missing keys:", missing)
    #print("unexpected keys:", unexpected)
    model.eval()
    return model

def export_onnx(
    checkpoint_path = "checkpoints/checkpoint_96.tar",
    onnx_path = "edgecnnmt.onnx",
    dummy_shape = (1,3,468),
    opset_version = 11
):
    model = load_model(checkpoint_path)
    wrapper = ONNXWrapper(model)
    dummy = torch.randn(*dummy_shape)

    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},
        opset_version=opset_version,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported and validated ONNX at {onnx_path}")
    return onnx_path, dummy
  

if __name__ == "__main__":
    export_onnx()










