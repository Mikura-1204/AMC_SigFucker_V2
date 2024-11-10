import torch
import torch.onnx
import os

# ciallo GPT: 从这里引进所需转换的模型类
from Models.CNN.cnn1d import ResNet1D

if __name__ == "__main__":
    # ciallo: cahnge your Fucking mode
    model = ResNet1D(5)

    # ciallo: Change this to your Fucking .pth-Path
    model_path = "/media/ubuntu/Elements/Ciallo_SigFucker_ShitType5/Results/experiment_TypeClassify_date_20241104190109/0_ResNet1D_A[1.0000]_L[0.0009].pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

    # 准备一个示例输入，形状应与模型的输入形状匹配
    input_shape = (32,2,2048)  # 假设输入形状为 
    dummy_input = torch.randn(input_shape)

    # ciallo: output path should also Fucking change
    output_path = "/media/ubuntu/Elements/Ciallo_SigFucker_ShitType5/Results/experiment_TypeClassify_date_20241104190109"
    output_name = os.path.join(output_path,'Fuck_Model.onnx')
    # 导出模型到 ONNX 文件
    torch.onnx.export(
        model,                # 要导出的模型
        dummy_input,          # 示例输入
        output_name,         # 输出文件名
        opset_version=11,     # ONNX 操作集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入节点的名称
        output_names=['output'],  # 输出节点的名称
        dynamic_axes={'input': {0: 'batch_size'},  # 动态轴
                    'output': {0: 'batch_size'}}
    )
    print("onnx文件转换完成喵 !",end='\n')