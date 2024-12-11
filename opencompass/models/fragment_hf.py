# flake8: noqa: E501
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from langchain_core.messages import ChatMessage

from opencompass.utils.prompt import PromptList
from FlowDesign.memory.base import faissStorage
from langchain_huggingface import HuggingFaceEmbeddings
from FlowDesign.chatbot.base import HuggingFacebot
from FlowDesign.flow.fragment import FragmentFlow 

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str, float]


class FragmentHF(BaseAPIModel):
    def __init__(
        self,
        hf_key: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        search_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        original_data_path: str = None,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        embeddings = HuggingFaceEmbeddings(model_name=search_model_name)
        if original_data_path is not None:
            self.db = faissStorage.load(name=original_data_path, embedding_function=embeddings)
        else:
            self.db = faissStorage(len(embeddings.embed_query("h")), embedding_function=embeddings)
        self.lm = HuggingFacebot(hf_key, model_repo_id=path)
        self.lm.system, self.lm.bot, self.lm.user = 'system', 'assistant', 'user'
        self.lm.clear_history()
        self.flow = FragmentFlow(self.db, self.lm)

    def _generate(self, input: PromptType):
        if isinstance(input, str):
            return self.flow.chat(input)['answer']
        else:
            messages = []
            for item in input:
                if item['role'] == 'HUMAN':
                    messages.append(ChatMessage(content=item['prompt'], role=self.lm.user))
                elif item['role'] == 'BOT':
                    messages.append(ChatMessage(content=item['prompt'], role=self.lm.bot))

            return self.flow.chat(messages)['answer']
            

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs))
        self.flush()
        return results