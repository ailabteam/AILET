# File: verify_gpu.py
import torch

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print("-" * 30)
    
    # Kiểm tra xem CUDA có sẵn không
    is_available = torch.cuda.is_available()
    print(f"CUDA available: {is_available}")
    
    if not is_available:
        print("!!! PyTorch không thể tìm thấy CUDA. GPU sẽ không được sử dụng. !!!")
        print("Hãy kiểm tra lại driver NVIDIA và cách bạn cài đặt PyTorch.")
        return

    print("-" * 30)
    
    # Lấy thông tin về các GPU
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s).")
    
    for i in range(device_count):
        print(f"\n--- Device {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"Total Memory: {total_mem:.2f} GB")

    # Thử tạo một tensor và chuyển nó lên GPU
    try:
        print("\n--- Testing Tensor Allocation ---")
        tensor = torch.randn(3, 3).to("cuda")
        print("Successfully created a tensor on cuda:0")
        print(tensor)
    except Exception as e:
        print(f"Error during tensor allocation on GPU: {e}")

if __name__ == "__main__":
    check_gpu()
