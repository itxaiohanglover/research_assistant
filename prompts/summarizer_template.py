# 导入必要的库和模块
from langchain.chains.llm import LLMChain  # 导入LLMChain，用于构建基于LLM的生成链
from langchain_core.prompts import PromptTemplate  # 导入PromptTemplate，用于构建提示模板


# 定义摘要模板
summarizer_template = """
假设你是一个AI科研助手，请用一段话概括下面文章的主要内容，200字左右。

{text}
"""

# 定义Summarizer类
class Summarizer:
    """
    Summarizer类用于将长文本内容压缩成简短的摘要。
    """

    def __init__(self, llm):
        self.llm = llm  # LLM实例，用于生成文本
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=summarizer_template
        )  # 定义摘要提示模板
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)  # 创建LLMChain实例，用于生成摘要

    def summarize(self, docs):
        """
        从文档中生成摘要。

        :param docs: 文档列表，其中每个文档包含page_content属性
        :return: 生成的摘要文本
        """
        # 从第一页中获取摘要内容，假设摘要位于'ABSTRACT'和'KEY WORDS'之间
        content = docs[0].page_content.split('ABSTRACT')[1].split('KEY WORDS')[0]
        summary = self.chain.run(content)  # 使用LLMChain生成摘要
        return summary  # 返回生成的摘要