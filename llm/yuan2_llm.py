# 导入必要的库
from typing import List, Optional
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
