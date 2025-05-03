from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .config import GeneratorConfig


QUERY_PROMPT = """
CONSIDERING THESE FACTS ABOUT USER:
{}
USER SAYS: {}
GIVE A RESPONSE TO THE USER's REPLIC:
"""


class ResponseGenerator:

  def __init__(self):

    self.tokenizer = AutoTokenizer.from_pretrained(GeneratorConfig.MODEL_NAME,)
    if self.tokenizer.pad_token_id is None:
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    self.model = AutoModelForCausalLM.from_pretrained(
       GeneratorConfig.MODEL_NAME,
       torch_dtype=torch.bfloat16,
       device_map=GeneratorConfig.DEVICE
    )
    self.pipeline = pipeline(task="text-generation", tokenizer=self.tokenizer, model=self.model)

    self.system_prompt = GeneratorConfig.SYSTEMPROMPT
    self.query_prompt = QUERY_PROMPT

    self.history = [{"role": "system", "content": self.system_prompt},]

  def gen_response(self, request:str, relevant_facts:list[str]) -> str:
    query_prompt = self.query_prompt.format("\n".join(relevant_facts), request)

    self.history.append({"role": "user", "content": query_prompt})

    templated = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
    output = self.pipeline(templated, max_new_tokens=300, return_full_text=False)
    result_txt = output[0].get('generated_text')

    self.history.append({"role": "assistant", "content": result_txt})

    return result_txt
