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