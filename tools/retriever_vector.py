"""
    向量检索器: 用于检索 order_faq 文件中的内容, 关于携程航班处理的规则
"""

import re
import numpy as np
from langchain.agents import create_agent
from langchain_core.tools import tool

from llm import embeddings_model, deepseek

# 读取 FAQ 文本文件
faq_text = None
with open('../order_faq.md', encoding='utf8') as f:
    faq_text = f.read()

# 将 FAQ 文本按标题分割成多个文档
docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


# 定义向量存储检索器类
class VectorStoreRetriever:
    """
        向量检索器类, 支持文档嵌入向量 以及 向量相似性检索
    """
    def __init__(self, docs: list, vectors: list):
        """
            构造方法
        :param docs: 文档内容列表, 每个文档都是一个dict
        :param vectors: 向量列表
        """
        self._arr = np.array(vectors)  # 向量列表转二维矩阵
        self._docs = docs              # 保存原始文档

    @classmethod
    def from_docs(cls, docs) :
        """
            工厂方法: 根据文本返回一个向量检索实例例
        :param docs: 待检索的文本
        :return: 向量检索实例
        """
        # 1、docs嵌入向量
        # for 循环遍历
        # embedding_docs = []
        # for doc in docs:
        #     embedding_doc = embeddings_model.embed_documents(doc["page_content"])
        #     embedding_docs.append(embedding_doc)
        # 列表生成器
        embedding_docs = embeddings_model.embed_documents([doc["page_content"] for doc in docs])

        # 2、返回向量检索实例
        vectors = embedding_docs
        return cls(docs, vectors)


    def query(self, query: str, k: int = 5) -> list[dict]:
        """
        query向量检索
        :param query: 检索query
        :param k: top-k
        :return: 检索结果, 字典列表
        """
        # 1、将query转换成查询向量
        embedding_query = embeddings_model.embed_query(query)
        # 2、计算查询向量与检索向量的相似度(使用点积即矩阵相乘, 归一化的向量等价于余弦相似度, 注意矩阵要转置)
        query_vectors = np.array(embedding_query)
        scores = query_vectors @ self._arr.T
        # 3、获取相似度最高的 k 个文档的索引
        top_k_index = np.argpartition(scores, -k)[-k:]
        top_k_index_sorted = top_k_index[np.argsort(-scores[top_k_index])]
        # 4、返回相似度 前top-k 个文档及其相似度

        # for循环拼装
        # result = []
        # for index in top_k_index_sorted:
        #     result.append(
        #         {
        #             **self._docs[index],         # 字典解包
        #             "similarity": scores[index],
        #         }
        #     )
        # return result

        # 列表生成器
        return [
            {**self._docs[index], "similarity": scores[index]} for index in top_k_index_sorted
        ]


# 创建dcos 的向量检索存储实例
retriever = VectorStoreRetriever.from_docs(docs)


# 定义工具函数，用于查询航空公司的政策
@tool
def lookup_policy(query: str) -> str:
    """
    查询公司政策，检查某些选项是否允许。
    在进行航班变更或其他'写'操作之前使用此函数。
    :param query: 查询问题
    :return: 匹配结果
    """
    # 查询相似度最高的 k 个文档
    docs = retriever.query(query, k=2)

    # 返回这些文档的内容
    return "\n\n".join([doc["page_content"] for doc in docs])


# 测试
if __name__ == '__main__':  # 测试代码
    agent = create_agent(
        model=deepseek,
        tools=[lookup_policy],
    )

    response = agent.invoke({"messages": [("user", "退票的手续是什么?")]})

    print(response)
    print(response["messages"][-1].content)
