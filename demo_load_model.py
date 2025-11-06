import torch
import sys
import os
import numpy as np
import json  # 新增：用于JSON文件处理
from PIL import Image
from transformers import AutoModel, AutoProcessor

def check_cuda():
    """Check CUDA availability and PyTorch installation"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check:")
        print("1. NVIDIA GPU is present")
        print("2. NVIDIA drivers are installed")
        print("3. PyTorch is installed with CUDA support")
        print("\nCurrent PyTorch setup:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        return False
    return True

# 递归处理嵌套字典，将所有张量移动到指定设备
def move_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj

# 新增：递归将NumPy数组转换为Python列表（用于JSON序列化）
def convert_numpy_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 将NumPy数组转为列表
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj  # 其他类型保持不变

# 修改：保存动作数据为JSON格式
def save_actions(actions, output_dir, filename):
    """
    保存动作数组到指定目录（JSON格式，支持VS Code直接打开）
    
    参数:
        actions: 包含动作数据的字典（含'actions'和'action_ids'数组）
        output_dir: 输出目录路径
        filename: 保存的文件名（无需加后缀）
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    # 拼接完整文件路径（添加.json后缀）
    output_path = os.path.join(output_dir, f"{filename}.json")
    
    # 转换数据：将NumPy数组转为Python列表（JSON可序列化）
    serializable_data = convert_numpy_to_list(actions)
    
    # 保存为JSON文件（indent=2确保格式化显示，增强可读性）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    print(f"动作数据已成功保存为JSON格式: {output_path}")

# 设置设备和数据类型
device = "cuda" if check_cuda() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

try:
    # 清理CUDA缓存
    torch.cuda.empty_cache()

    model_name_or_path = "hf_models/spatialvla-4b-224-pt"
    processor = AutoProcessor.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True, 
        torch_dtype=dtype,
        device_map="auto"
    ).eval().to(device)

    # 加载输入图像和提示文本
    image = Image.open("example.png").convert("RGB")
    prompt = "What action should the robot take to pick the cup?"
    
    # 处理输入并移动到目标设备
    inputs = processor(images=[image], text=prompt, return_tensors="pt")
    inputs = move_to_device(inputs, device)

    # 模型推理生成动作
    generation_outputs = model.predict_action(inputs)
    actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
    print("生成的动作数据:")
    print(actions)

    # 保存动作数据（JSON格式）
    output_directory = "output"
    output_filename = "demo_action_output"  # 无需加后缀，函数内会自动添加.json
    save_actions(actions, output_directory, output_filename)

except Exception as e:
    print(f"Error: {e}")
    print("\nTo install PyTorch with CUDA support, run:")
    print("conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    sys.exit(1)