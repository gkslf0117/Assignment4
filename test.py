import os
import sys
import subprocess
import torch
import torch.nn as nn

# 과제3 동일 외부 모델 정의 (784 -> 16 -> 10 MLP)
class TinyMNIST(nn.Module):
    def __init__(self):
        super(TinyMNIST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def export_to_onnx(model_path="mnist_small.onnx"):
    """과제 #3과 동일한 구조의 외부 ONNX 모델 파일 생성 함수"""
    print(f"[*] Exporting TinyMNIST to {model_path}...")
    model = TinyMNIST()
    model.eval()
    
    # MNIST 표준 입력 차원 
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model, 
        dummy_input, 
        model_path, 
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("[+] ONNX model export complete.")

def run_alpha_beta_crown():
    """터미널 명령어 안전 실행"""
    print("[*] Launching 𝛼,𝛽-Crown verification via subprocess...")
    
    verifier_script = os.path.join(".", "alpha-beta-CROWN", "complete_verifier", "abcrown.py")
    
    if not os.path.exists(verifier_script):
        print("[-] Error: 𝛼,𝛽-Crown repository not found at ./alpha-beta-CROWN.")
        print("    Please run 'git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git' first.")
        return

    cmd = [sys.executable, verifier_script, "--config", "mnist_verification.yaml"]
    
    try:
        subprocess.run(cmd, check=True)
        print("[+] Verification finished successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[-] Verification failed with exit code {e.returncode}")
    except Exception as e:
        print(f"[-] Unexpected error occurred: {e}")

if __name__ == "__main__":
    # 과제 #3과 동일한 ONNX 모델을 생성
    export_to_onnx()
        
    # 𝛼,𝛽-Crown 검증 실행
    run_alpha_beta_crown()
