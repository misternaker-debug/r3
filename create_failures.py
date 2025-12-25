import torch
import datasets
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, concatenate_datasets
import data_processed
import sys
import os
from trl.import_utils import is_math_verify_available
from sentence_transformers import SentenceTransformer
#model_judge = SentenceTransformer('all-MiniLM-L6-v2')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    
class FailureDatasetCreator:
    """
    Создает датасет ошибок для тренировки по методике R³
    """
    
    def __init__(
        self,
        base_model: str,
        use_vllm: bool = True,
        vllm_tensor_parallel_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.use_vllm = use_vllm
        
        if use_vllm:
            # Инициализация vLLM для быстрой генерации
            self.llm = LLM(
                model=base_model,
                tensor_parallel_size=vllm_tensor_parallel_size,
                gpu_memory_utilization=0.8,
                max_num_seqs=256,
                trust_remote_code=True
            )
        else:
            # Стандартная инициализация модели
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch.float16
                ).bfloat16().cuda()
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
                self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
                self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    
    def generate_responses(
        self,
        prompts: List[str],
        num_responses_per_prompt: int = 64,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, List[List[str]]]:
        """
        Генерирует несколько ответов для каждого промпта
        """
        all_responses = {}
        
        if self.use_vllm:
            # Генерация с помощью vLLM с префиксным кэшированием
            sampling_params = SamplingParams(
                n=num_responses_per_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                skip_special_tokens=True
            )
            
            # Повторяем каждый промпт num_responses_per_prompt раз
            repeated_prompts = []
            for prompt in prompts:
                repeated_prompts.extend([prompt] * num_responses_per_prompt)
            
            # Генерация ответов
            outputs = self.llm.generate(repeated_prompts, sampling_params)
            
            # Группируем ответы по промптам
            for i, prompt in enumerate(prompts):
                start_idx = i * num_responses_per_prompt
                end_idx = start_idx + num_responses_per_prompt
                prompt_responses = []
                
                for j in range(start_idx, end_idx):
                    if j < len(outputs):
                        response = outputs[j].outputs[0].text
                        prompt_responses.append(response)
                
                all_responses[prompt] = prompt_responses
                
        else:
            # Стандартная генерация с помощью transformers
            for prompt in tqdm(prompts, desc="Generating responses"):
                prompt_responses = []
                
                # Токенизация промпта
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                for _ in range(num_responses_per_prompt):
                    # Генерация ответа
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Декодирование ответа
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:]
                    )
                    prompt_responses.append(response)
                
                all_responses[prompt] = prompt_responses
        
        return all_responses

    def check_answer(self, response, answer):
        reasoning_delimiters = ["</think>"]
        return True if self.reasoning_accuracy_reward(response, [answer], reasoning_delimiters=reasoning_delimiters) == 1.0 else False

    def reasoning_accuracy_reward(
        self,
        completions: list[list[dict[str, str]]],
        solution: list[str],
        reasoning_delimiters: list[str] | None = None,
        **kwargs,
    ) -> list[float | None]:

        if not is_math_verify_available():
            raise ImportError("Please install the `math_verify` package to use reasoning_accuracy_reward")

        if reasoning_delimiters is None:
            # Use sensible defaults for majority of reasoning models
            reasoning_delimiters = ["</think>"]

        rewards = []
        contents = [completions]
        for content, sol in zip(contents, solution, strict=True):
            # Split final answer from reasoning content
            is_reasoning_complete = False
            for delim in reasoning_delimiters:
                if delim in content:
                    content = content.split(delim)[-1]
                    is_reasoning_complete = True
                    break
            if not is_reasoning_complete:
                # We assign zero reward instead of `None` to penalize incomplete reasoning
                rewards.append(0.0)
                continue

            gold_parsed = parse(sol)
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            boxed_match_priority=0,
                            normalization_config=NormalizationConfig(
                                units=True,
                            ),
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            else:
                # If the gold solution cannot be parsed, we assign `None` to skip this example
                reward = 0
            rewards.append(reward)

        return rewards

    def verify_responses(
        self,
        original_dataset: Dataset,
        all_responses: Dict[str, List[str]],
    ) -> Dict[str, List[bool]]:
        """
        Проверяет правильность сгенерированных ответов
        """
        verification_results = {}
        i = 0
        for key in all_responses.keys():
            prompt_responses = all_responses[key]
            correctness_flags = []
            for response in prompt_responses:
                is_correct = self.check_answer(response, original_dataset['solution'][i])
                
                correctness_flags.append(is_correct)
            
            verification_results[key] = correctness_flags
            i += 1
        
        return verification_results
    
    def semantic_similarity(self, response, answer):
        treshold = 0.9
        embedding_res = model_judge.encode(response)
        embedding_ans = model_judge.encode(answer)
        print('RESPONSE',response,'/n', 'ANSWER', answer, '/n', cosine_similarity([embedding_res], [embedding_ans]))
        return True if cosine_similarity([embedding_res], [embedding_ans]) >= treshold else False
    
    def create_failure_dataset(
        self,
        original_dataset: Dataset,
        prompt_column: str = "formatted",
        num_responses_per_prompt: int = 64
    ) -> Dataset:
        """
        Создает датасет ошибок из оригинального датасета
        """
        # Извлекаем промпты
        prompts = original_dataset[prompt_column]
        
        logging.info(f"Processing {len(prompts)} prompts...")
        # Генерируем ответы для каждого промпта
        all_responses = self.generate_responses(
            prompts, 
            num_responses_per_prompt=num_responses_per_prompt
        )
    
        # Проверяем ответы
        verification_results = self.verify_responses(
            original_dataset, 
            all_responses, 
        )
        # Собираем датасет ошибок
        failure_data = []
        
        for i, prompt in enumerate(prompts):
            responses = all_responses.get(prompt, [])
            correctness = verification_results.get(prompt, [])

            for j, (response, is_correct) in enumerate(zip(responses, correctness)):
                    if not is_correct:  
                        new_prompt = f"""Вы — высококвалифицированный аналитик, перед которым поставлена задача провести углубленный анализ ошибки в ответе. Ваша задача - понять, почему ответ неверен.:
                        {response} был сгенерирован, несмотря на первоначальный запрос: {prompt}.
        Проанализируйте свои действия и разработайте план по улучшению будущих ответов."""
                        failure_data.append({
                            "prompt": new_prompt,
                            "solution": original_dataset['solution'][i]
                        })
                        
        logging.info(f"Created failure dataset with {len(failure_data)} samples")
        logging.info(f"Failure rate: {len(failure_data) / (len(prompts) * num_responses_per_prompt):.2%}")
        dataset = Dataset.from_list(failure_data)
        dataset = dataset.map(lambda x: {
                                    "prompt" : [
                                    {"role": "user", "content": x['prompt']}
                                    ],
                                    "solution": x["solution"],
                                })
        return dataset
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    file_path = sys.argv[1]
    model_name = sys.argv[2]
    num_responses_per_prompt = int(sys.argv[3])
    creator = FailureDatasetCreator(
        base_model=model_name,
        use_vllm=True
    )
    data = data_processed.process_data(file_path, model_name)
    failure_dataset = creator.create_failure_dataset(
        data,
        num_responses_per_prompt=num_responses_per_prompt
      
    )
    failure_dataset = failure_dataset.to_pandas()

    failure_dataset.to_json("failures_dataset.json",  orient='records', lines=False, force_ascii=False, indent=4)
