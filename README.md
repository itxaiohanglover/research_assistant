### 实战 - AI科研助手
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/139c75addae84c58a82ac54357688f71.png#pic_center)

项目主要包含一个Streamlit开发的客户端，以及一个部署好的浪潮源大模型的服务端。​
- 客户端接收到用户上传的PDF后，发送到服务端。服务端首先完成PDF内容解析，然后拼接摘要Prompt并输入源大模型，得到模型输出结果后，返回给客户端并展示给用户。​
- 如果用户接下来进行提问，客户端将用户请求发送到服务端，服务端进行Embedding和Faiss检索，然后将检索到的chunks与用户请求拼接成Prompt并输入到源大模型，得到模型输出结果后，返回给客户端进行结构化，然后展示给用户。

#### 项目结构
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b60ac1a9f3a84eeb81649234d45e4d7f.png#pic_center)
主模块：`main.py`
```python
# 导入必要的库和模块
from langchain_community.document_loaders import PyPDFLoader  # 用于加载PDF文件的加载器
from common import constants  # 导入常量配置
import streamlit as st  # 导入Streamlit库，用于构建Web界面

from llm.yuan2_llm import Yuan2_LLM  # 导入自定义的大型语言模型类
from langchain_huggingface import HuggingFaceEmbeddings  # 导入HuggingFace嵌入模型
# 导入提示模板类
from prompts.chatbot_template import ChatBot
from prompts.summarizer_template import Summarizer

# 定义模型路径
model_path = constants.MODEL_PATH  # 从常量配置中获取模型路径

# 定义向量模型路径
embedding_model_path = constants.EMBED_MODEL_PATH  # 从常量配置中获取向量模型路径


# 定义一个函数，用于获取llm和embeddings
@st.cache_resource  # 使用Streamlit的缓存装饰器来缓存函数的结果
def get_models():
    llm = Yuan2_LLM(model_path)  # 创建Yuan2_LLM实例

    # 定义模型和编码的参数
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}  # 设置为True以计算余弦相似度
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,  # 向量模型的名称或路径
        model_kwargs=model_kwargs,  # 模型参数
        encode_kwargs=encode_kwargs,  # 编码参数
    )
    return llm, embeddings  # 返回创建的LLM和嵌入模型实例


def main():
    # 创建一个标题
    st.title('💬 Yuan2.0 AI科研助手')  # 设置Streamlit应用的标题

    # 获取llm和embeddings
    llm, embeddings = get_models()  # 调用get_models函数获取模型实例

    # 初始化summarizer
    summarizer = Summarizer(llm)  # 创建Summarizer实例用于生成文本摘要

    # 初始化ChatBot
    chatbot = ChatBot(llm, embeddings)  # 创建ChatBot实例用于回答问题

    # 上传pdf
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')  # 创建文件上传器，允许用户上传PDF文件

    if uploaded_file:
        # 加载上传PDF的内容
        file_content = uploaded_file.read()  # 读取上传的文件内容

        # 写入临时文件
        temp_file_path = "temp.pdf"  # 定义临时文件路径
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)  # 将文件内容写入临时文件

        # 加载临时文件中的内容
        loader = PyPDFLoader(temp_file_path)  # 创建PDF加载器实例
        docs = loader.load()  # 使用加载器加载文档内容

        st.chat_message("assistant").write(f"正在生成论文概括，请稍候...")  # 在Streamlit界面上显示消息

        # 生成概括
        summary = summarizer.summarize(docs)  # 调用summarizer的summarize方法生成摘要

        # 在聊天界面上显示模型的输出
        st.chat_message("assistant").write(summary)  # 显示生成的摘要

        # 接收用户问题
        if query := st.text_input("Ask questions about your PDF file"):  # 创建文本输入框，允许用户输入问题
            # 检索 + 生成回复
            chunks, response = chatbot.run(docs, query)  # 调用chatbot的run方法进行检索和生成回答

            # 在聊天界面上显示模型的输出
            st.chat_message("assistant").write(f"正在检索相关信息，请稍候...")  # 显示检索信息的消息
            st.chat_message("assistant").write(chunks)  # 显示检索到的文档片段

            st.chat_message("assistant").write(f"正在生成回复，请稍候...")  # 显示生成回答的消息
            st.chat_message("assistant").write(response)  # 显示生成的回答


if __name__ == '__main__':
    main()  # 如果是主程序，则调用main函数运行应用
```
提示词模块：

`chatbot_template.py`

```python
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入文本分割器
from langchain.chains.question_answering import load_qa_chain  # 导入load_qa_chain，用于加载问答链
from langchain_community.vectorstores import FAISS
# 定义聊天机器人模板
chatbot_template = '''
假设你是一个AI科研助手，请基于背景，简要回答问题。

背景：
{context}

问题：
{question}
'''.strip()


# 定义ChatBot类
class ChatBot:
    """
    ChatBot类用于处理用户提问，并基于文档内容生成回答。
    """

    def __init__(self, llm, embeddings):
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=chatbot_template
        )  # 定义聊天机器人提示模板
        self.chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=self.prompt)  # 加载问答链
        self.embeddings = embeddings  # 嵌入模型，用于文档向量化

        # 加载文本分割器，用于将长文本切分成小块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=10,
            length_function=len
        )

    def run(self, docs, query):
        """
        处理用户提问，生成回答。

        :param docs: 文档列表，每个文档包含page_content属性
        :param query: 用户的提问
        :return: 检索到的文档片段和生成的回答
        """
        # 读取所有文档内容
        text = ''.join([doc.page_content for doc in docs])

        # 使用文本分割器切分成chunks
        all_chunks = self.text_splitter.split_text(text=text)

        # 将文本chunks转换为向量并存储
        VectorStore = FAISS.from_texts(all_chunks, embedding=self.embeddings)

        # 检索与提问最相似的chunks
        chunks = VectorStore.similarity_search(query=query, k=1)

        # 使用问答链生成回答
        response = self.chain.run(input_documents=chunks, question=query)

        return chunks, response  # 返回检索到的文档片段和生成的回答

```
`summarizer_template.py`
```python
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
```
源大模型模块：
`yuan2_llm.py`
```python
# 导入必要的库
from typing import List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 导入常量配置
from common import constants

# 导入LLM基类
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# 定义模型路径
model_path = constants.MODEL_PATH

# 定义模型数据类型
torch_dtype = torch.bfloat16


# 定义源大模型类
class Yuan2_LLM(LLM):
    """
    YUAN2_LLM类用于加载和使用预训练的大型语言模型。
    它继承自langchain的LLM基类，并实现了自己的_call方法来生成文本。
    """

    # 类变量，用于存储分词器和模型实例
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str):
        super().__init__()

        # 加载预训练的分词器和模型
        print("Creating tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False,
                                                       eos_token='<eod>')
        # 添加特殊标记
        self.tokenizer.add_tokens(
            ['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
             '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>',
             '<jupyter_output>', '<empty_output>'], special_tokens=True)

        print("Creating model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype,
                                                          trust_remote_code=True).cuda()

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """
        生成文本的方法，根据输入的prompt生成响应。

        :param prompt: 输入的文本提示
        :param stop: 停止生成的标记列表
        :param run_manager: 运行管理器，用于监控和管理生成过程
        :param kwargs: 其他可选参数
        :return: 生成的文本响应
        """
        prompt = prompt.strip()
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(inputs, do_sample=False, max_length=4096)
        output = self.tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1].split("<eod>")[0]

        return response

    @property
    def _llm_type(self) -> str:
        """
        返回模型的类型标识。

        :return: 模型类型字符串
        """
        return "Yuan2_LLM"
```
#### 运行效果
```shell
bash run.sh
```
启动脚本：`run.sh`
```shell
streamlit run main.py --server.address 127.0.0.1 --server.port 6006
```
论文概括：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b90be2cc1a09497a97b688026c8255d8.png#pic_center)
```
What kind of attention architecture is LFA?
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e130415e604e4f05be3f84586c374353.png#pic_center)