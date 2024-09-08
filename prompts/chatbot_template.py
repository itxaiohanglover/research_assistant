from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入文本分割器
from langchain.chains.question_answering import load_qa_chain  # 导入load_qa_chain，用于加载问答链

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
