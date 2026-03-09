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
import re
from trl.import_utils import is_math_verify_available
from sentence_transformers import SentenceTransformer
from typing import Tuple, Union, Optional

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
        solutions: List[str],
        num_responses_per_prompt: int = 64,
        max_tokens: int = 2000,
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
                solution = solutions[i]
                for j in range(start_idx, end_idx):
                    if j < len(outputs):
                        response = outputs[j].outputs[0].text
                        prompt_responses.append(response)
                
                all_responses[tuple([prompt, solution])] = prompt_responses
                
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

    def extract_answer_from_content(self, content: str) -> str:
        """
        Извлекает ответ из контента модели.
        
        Args:
            content: Текст ответа модели, может содержать теги <answer>
        
        Returns:
            Извлеченный ответ (очищенный текст)
        """
        if not content:
            return ""
        
        # 1. Пытаемся извлечь содержимое тега <answer>
        
        answer_patterns = [
            r'<answer>(.*?)</answer>',  # с тегами
            r'answer is ([A-E])',       # "answer is B"
            r'answer: ([A-E])',         # "answer: B"
            r'Answer: ([A-E])',         # "Answer: B"
            r'Ответ: ([A-E])',          # "Ответ: B"
            r'correct answer is ([A-E])', # "correct answer is B"
            r'правильный ответ ([A-E])'  # "правильный ответ B"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        
            if match:
                extracted = match.group(1).strip()  
                if extracted in ['A', 'B', 'C', 'D', 'E']:
                    return extracted
        
        # 2. Если не нашли по паттернам, ищем одиночную букву A-E
        letter_match = re.search(r'\b([A-E])\b', content, re.IGNORECASE)
        if letter_match:
            if letter_match in ['A', 'B', 'C', 'D', 'E']:
                return letter_match
            return letter_match.group(1).upper()
        
        # 3. Возвращаем очищенный текст (убираем теги)
        cleaned = re.sub(r'<[^>]+>', '', content).strip()

        return cleaned

    def normalize_answer(self, answer: str) -> str:
        """
        Нормализует ответ для сравнения.
        
        Args:
            answer: Ответ (модели или ground truth)
        
        Returns:
            Нормализованный ответ
        """
        if not answer:
            return ""
        
        # Приводим к верхнему регистру
        answer = str(answer).upper().strip()
        
        # Убираем лишние пробелы
        answer = re.sub(r'\s+', ' ', answer)
        
        # Если ответ содержит несколько вариантов через запятую, берем первый
        if ',' in answer:
            parts = [p.strip() for p in answer.split(',')]
            return parts[0]
        
        return answer

    def reward_function(
        self,
        content: str,
        ground_truth: str,
        only_reward: bool,
    ) -> Tuple[float, str, str]:
        """
        Вычисляет награду за ответ модели.
        
        Args:
            content: Ответ модели (может содержать теги)
            ground_truth: Правильный ответ

        Returns:
            Tuple: (reward, extracted_answer, matched_part)
        """
        
        # 1. Извлекаем ответ из контента модели
        extracted_answer = self.extract_answer_from_content(content)

        # 2. Нормализуем оба ответа
        normalized_model_answer = self.normalize_answer(extracted_answer)
        normalized_ground_truth = self.normalize_answer(ground_truth)
        
        # 3. Проверяем точное совпадение для буквенных ответов
        if normalized_ground_truth in ['A', 'B', 'C', 'D', 'E']:
            # Для буквенных ответов всегда используем точное совпадение
            is_correct = (normalized_model_answer == normalized_ground_truth)
            reward = 1.0 if is_correct else 0.0
            if only_reward:
                return reward
            return reward, extracted_answer, normalized_model_answer
        if only_reward:
            return 0.0
        return 0.0, extracted_answer, normalized_model_answer

    def check_answer(self, response, answer):
        return True if self.reward_function(response, answer, True) == 1.0 else False


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
            _, solution = key
            correctness_flags = []
            for response in prompt_responses:
                #match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                #if match:
                 #   response = match.group(1).strip()
                   # if len(ans) >= 1:
                        #response = ans
                is_correct = self.check_answer(response, solution)
                correctness_flags.append(is_correct)
            
            verification_results[key] = correctness_flags
            i += 1
        
        return verification_results
    
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
        solutions = original_dataset["solution"]
        logging.info(f"Processing {len(prompts)} prompts...")
        # Генерируем ответы для каждого промпта
        all_responses = self.generate_responses(
            prompts, 
            solutions,
            num_responses_per_prompt=num_responses_per_prompt
        )
        
        # Проверяем ответы
        verification_results = self.verify_responses(
            original_dataset, 
            all_responses, 
        )
        # Собираем датасет ошибок
        failure_data = []
        for i, (prompt, solution) in enumerate(zip(prompts, solutions)):
            responses = all_responses.get(tuple([prompt,solution]), [])
            correctness = verification_results.get(tuple([prompt,solution]), [])

            for j, (response, is_correct) in enumerate(zip(responses, correctness)):
                    if not is_correct:  
                        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                        if match:
                            response = match.group(1).strip()
                        new_prompt = f"""Вы — высококвалифицированный аналитик, перед которым поставлена задача провести 
                        углубленный анализ ошибки в ответе. Ваша задача - понять, почему твой прошлый ответ оказался неверен:{response}. 
                        Первоначальный запрос: {prompt}.
                        Проанализируйте свои действия и разработайте план по улучшению будущих ответов. В ходе размышлений 
                        необходимо учесть тот факт, что нельзя выбрать сразу несколько вариантов ответов"""
                        failure_data.append({
                            "prompt": new_prompt,
                            "solution": solution,
                            "original_query": prompt
                        })
                        
        logging.info(f"Created failure dataset with {len(failure_data)} samples")
        logging.info(f"Failure rate: {len(failure_data) / (len(prompts) * num_responses_per_prompt):.2%}")
        dataset = Dataset.from_list(failure_data)
        dataset = dataset.map(lambda x: {
                                    "prompt" : [
                                    {"role": "user", "content": x['prompt']}
                                    ],
                                    "solution": x["solution"],
                                    "original_query": x["original_query"]
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

    failure_dataset.to_json("failures_dataset.json",  orient='records', lines=False, force_ascii=False, indent=5)
