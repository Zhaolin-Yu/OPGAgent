"""
DentalGPT API 客户端调用示例
在局域网的另一台机器上运行此脚本来调用 API
"""

import requests
import base64
import json
from pathlib import Path


# ========== 配置 ==========
# 将此处的 IP 地址改为服务器的局域网 IP
SERVER_IP = "192.168.1.100"  # 修改为你的服务器 IP
SERVER_PORT = 8566
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"


def check_health():
    """检查服务器健康状态"""
    print("=" * 60)
    print("检查服务器健康状态...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 服务器状态: {result['status']}")
            print(f"✓ 模型已加载: {result['model_loaded']}")
            print(f"✓ 设备: {result['device']}")
            print(f"✓ 时间戳: {result['timestamp']}")
            return True
        else:
            print(f"✗ 健康检查失败: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ 无法连接到服务器: {e}")
        print(f"请确认:")
        print(f"  1. 服务器已启动")
        print(f"  2. IP 地址正确: {SERVER_IP}")
        print(f"  3. 防火墙已开放端口: {SERVER_PORT}")
        return False


def analyze_image_file(image_path, question=None):
    """
    使用文件上传方式分析图像
    
    参数:
        image_path: 本地图像文件路径
        question: 要问的问题（可选）
    """
    print("\n" + "=" * 60)
    print(f"方法 1: 文件上传方式分析图像")
    print("=" * 60)
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"✗ 图像文件不存在: {image_path}")
        return None
    
    print(f"图像路径: {image_path}")
    
    # 准备请求数据
    files = {
        'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    data = {}
    if question:
        data['question'] = question
        print(f"问题: {question}")
    
    try:
        print("\n发送请求到服务器...")
        response = requests.post(
            f"{BASE_URL}/analyze",
            files=files,
            data=data,
            timeout=300  # 5分钟超时
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ 分析成功!")
            print(f"\n问题: {result['question']}")
            print(f"\n回答:\n{result['answer']}")
            print(f"\n时间戳: {result['timestamp']}")
            return result
        else:
            print(f"\n✗ 分析失败: HTTP {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("\n✗ 请求超时，模型推理时间较长，请稍后再试")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        return None
    finally:
        files['image'][1].close()


def analyze_image_base64(image_path, question=None):
    """
    使用 Base64 编码方式分析图像
    
    参数:
        image_path: 本地图像文件路径
        question: 要问的问题（可选）
    """
    print("\n" + "=" * 60)
    print(f"方法 2: Base64 编码方式分析图像")
    print("=" * 60)
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"✗ 图像文件不存在: {image_path}")
        return None
    
    print(f"图像路径: {image_path}")
    
    # 读取并编码图像
    print("正在编码图像...")
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 准备请求数据
    data = {
        'image_base64': image_base64
    }
    
    if question:
        data['question'] = question
        print(f"问题: {question}")
    
    try:
        print("\n发送请求到服务器...")
        response = requests.post(
            f"{BASE_URL}/analyze/base64",
            data=data,
            timeout=300  # 5分钟超时
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ 分析成功!")
            print(f"\n问题: {result['question']}")
            print(f"\n回答:\n{result['answer']}")
            print(f"\n时间戳: {result['timestamp']}")
            return result
        else:
            print(f"\n✗ 分析失败: HTTP {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("\n✗ 请求超时，模型推理时间较长，请稍后再试")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        return None


def main():
    """主函数 - 演示如何调用 API"""
    
    print("\n" + "=" * 60)
    print("DentalGPT API 客户端示例")
    print("=" * 60)
    print(f"服务器地址: {BASE_URL}")
    print(f"API 文档: {BASE_URL}/docs")
    
    # 1. 健康检查
    if not check_health():
        return
    
    # 2. 示例图像路径（请修改为你的图像路径）
    example_image = "dental_image.jpg"
    
    # 示例问题
    example_question = "请分析图像中有问题的牙齿,并给出有问题的牙齿所在位置,牙齿类型"
    
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print("""
    请将下面代码中的参数修改为实际值:
    
    1. SERVER_IP: 修改为服务器的局域网 IP
    2. example_image: 修改为你要分析的图像文件路径
    3. example_question: 修改为你想问的问题（可选）
    
    然后运行下面的函数:
    """)
    
    # 取消下面的注释来运行示例
    # result1 = analyze_image_file(example_image, example_question)
    # result2 = analyze_image_base64(example_image, example_question)
    
    print("\n提示: 取消注释上面的代码行来测试 API 调用")


if __name__ == "__main__":
    main()
