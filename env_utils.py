"""
    环境变量导入工具
"""
import os
from dotenv import load_dotenv

# override=True 确保.env文件优先
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
ALIYUN_BASE_URL = os.getenv("ALIYUN_BASE_URL")