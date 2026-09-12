"""
    模型初始化
"""
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings

from env_utils import DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 1、创建deepseek LLM实例
deepseek = init_chat_model(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-v4-flash",  # 模型名称
    model_provider="deepseek",  # 模型供应商
    temperature=0.9,  # 温度调节器
    max_tokens=10000,  # 最大输出token数
)


# 2、创建嵌入模型实例
embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)
