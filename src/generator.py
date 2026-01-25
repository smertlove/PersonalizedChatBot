from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .config import GeneratorConfig
from vllm import LLM, SamplingParams


QUERY_PROMPT = """
CONSIDERING THESE FACTS ABOUT USER:
{}
USER SAYS: {}
GIVE A RESPONSE TO THE USER's REPLIC:
"""


class ResponseGenerator:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            GeneratorConfig.MODEL_NAME,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            GeneratorConfig.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=GeneratorConfig.DEVICE,
            attn_implementation="eager",
            use_flash_attention_2=False,
        )
        self.pipeline = pipeline(
            task="text-generation", tokenizer=self.tokenizer, model=self.model
        )

        self.system_prompt = GeneratorConfig.SYSTEMPROMPT
        self.query_prompt = QUERY_PROMPT


        self.pairs = 6
        self.history = [{"role": "system", "content": self.system_prompt}]

    @torch.no_grad()
    def gen_response(self, request: str, relevant_facts: list[str]) -> str:
        # torch.cuda.empty_cache()

        # query_prompt = request
        query_prompt = self.query_prompt.format(
            "\n".join(relevant_facts),
            request
        )

        self.history.append({"role": "user"  , "content": query_prompt})

        templated = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        output = self.pipeline(templated, max_new_tokens=300, return_full_text=False)

        result_txt = output[0].get("generated_text")

        self.history = self.history[:-1] + [
            {"role": "user"  , "content": request},
            {"role": "assistant", "content": result_txt}
       ]
    
        if len(self.history) - 1 > self.pairs * 2:
            self.history = [self.history[0]] + self.history[3:]

        return result_txt

    def response(self, request):
        return self.gen_response(request, None)


class ResponseGenerator_vLLM:
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            GeneratorConfig.MODEL_NAME,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = LLM(
            model=GeneratorConfig.MODEL_NAME,
            tokenizer=GeneratorConfig.MODEL_NAME,
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.8,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=False,
            tensor_parallel_size=1,
            max_model_len=2048,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            max_tokens=300,
            temperature=0.6,
        )

        self.system_prompt = GeneratorConfig.SYSTEMPROMPT
        self.query_prompt = QUERY_PROMPT

        self.pairs = 6
        self.history = [{"role": "system", "content": self.system_prompt}]

    @torch.no_grad()
    def gen_response(self, request: str, relevant_facts: list[str]) -> str:

        if relevant_facts:
            query_prompt = self.query_prompt.format(
                "\n".join(relevant_facts),
                request
            )
        else:
            query_prompt = request

        self.history.append({"role": "user"  , "content": query_prompt})

        templated = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        outputs = self.llm.generate(
            prompts=[templated],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        result_txt = outputs[0].outputs[0].text.strip()

        self.history = self.history[:-1] + [
            {"role": "user"  , "content": request},
            {"role": "assistant", "content": result_txt}
       ]
    
        if len(self.history) - 1 > self.pairs * 2:
            self.history = [self.history[0]] + self.history[3:]

        return result_txt