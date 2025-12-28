import torch
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
import logging
import numpy as np
import sys
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward
from trl import GRPOConfig
import torch
import copy
from typing import Any
from trl.trainer.utils import (
split_pixel_values_by_grid,
shuffle_sequence_dict,
split_tensor_dict,
unsplit_pixel_values_by_grid,
)
from peft import LoraConfig, get_peft_model
import os
import re
#from trl.chat_template_utils import add_response_schema, get_training_chat_template, parse_response
import transformers
from packaging.version import Version
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
    prepare_multimodal_messages_vllm,
)
from peft import LoraConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_bitsandbytes_available,
    is_trackio_available,
    is_wandb_available,
)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from trl.import_utils import is_math_verify_available


if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify


def extract_answer_from_content(content: str) -> str:
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

def normalize_answer(answer: str) -> str:
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
    extracted_answer = extract_answer_from_content(content)

    # 2. Нормализуем оба ответа
    normalized_model_answer = normalize_answer(extracted_answer)
    normalized_ground_truth = normalize_answer(ground_truth)
    
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


class R3GRPOTrainer(GRPOTrainer):
    """
    R³ (Reflect, Retry, Reward) GRPOTrainer с модификацией _prepare_inputs
    для реализации трехэтапного процесса:
    1. Генерация самоанализа (reflection) - k вариантов на промпт
    2. Генерация финального ответа (second_step) на основе каждого самоанализа
    3. Обновление RL политики для генерации самоанализа
    """
    
    def __init__(
        self,
        *args,
        prompts: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Функции для R³ процесса
        self.second_step_func = self._default_second_step
        
        
        # Кэш для second_step результатов
        self._second_step_cache = {}
        
        # Статистика
        self.stats = {
            'total_reflections': 0,
            'successful_answers': 0,
            'failed_answers': 0
        }
        
        # Настройки генерации для второго шага
        self.second_step_generation_config = {
            'max_new_tokens': 400,
            'temperature': 0.6,
            'top_p': 0.95,
            'do_sample': True,
            'pad_token_id': self.processing_class.pad_token_id,
            'eos_token_id': self.processing_class.eos_token_id,
            'bos_token_id': self.processing_class.bos_token_id
        }
        
        # Флаг для отслеживания первого прохода
        self._first_pass = True
        
        logging.info("R³ GRPOTrainer initialized with second_step functionality")
    
    def _extract_original_query(self, reflection_prompt: str) -> str:
        """
        Извлекает оригинальный запрос из промпта самоанализа.
        Формат промпта: "Original Query: ...\nFirst Failed Attempt: ...\nInstruction: ..."
        """
        import re
        
        patterns = [
            r"Original Query:\s*(.*?)(?:\n|$)",
            r"Question:\s*(.*?)(?:\n|$)",
            r"Запрос:\s*(.*?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reflection_prompt, re.IGNORECASE | re.DOTALL)
            if match:
                query = match.group(1).strip()
                # Ограничиваем длину
                if len(query) > 200:
                    query = query[:200] + "..."
                return query
        
        # Если не нашли, возвращаем первые 200 символов промпта
        return reflection_prompt[:200] + "..." if len(reflection_prompt) > 200 else reflection_prompt

    def _decode_completions(self, completion_ids: torch.Tensor) -> List[str]:
        """
        Декодирует completion_ids в текстовые completions.
        """
        completions = []
        batch_size = completion_ids.size(0)
        
        for i in range(batch_size):
            # Получаем последовательность токенов для i-го примера
            tokens = completion_ids[i]
            
            # Удаляем паддинг токены
            non_pad_mask = tokens != self.pad_token_id
            non_pad_tokens = tokens[non_pad_mask]
            
            # Декодируем в текст
            text = self.processing_class.decode(non_pad_tokens, skip_special_tokens=True)
            completions.append(text)
        
        return completions
    
    def _extract_reflection_from_completion(self, completion: str) -> str:
        """
        Извлекает часть самоанализа из completion.
        В completion может быть только самоанализ или самоанализ + что-то еще.
        """
        # Ищем теги <think>
        import re
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        
        # Если нет тегов, возвращаем весь текст (предполагаем, что это чистый самоанализ)
        return completion.strip()
    
    def _default_second_step(self, original_query: str, reflection: str, model=None) -> str:
        """
        Функция второго шага по умолчанию.
        Генерирует финальный ответ на основе оригинального запроса и самоанализа.
        """
        # Создаем промпт для генерации финального ответа
        answer_prompt = f"""Основываясь на приведенных ниже выводах, дайте правильный ответ на исходный запрос.

Исходный запрос: {original_query}

Размышления о предыдущей попытке: {reflection}

Теперь дайте правильный ответ из предоженных вариантов: A, B, C, D
"""
        
        # Используем предоставленную модель или self.model

        # Генерация финального ответа (вне графа вычислений)
        self.model.eval()
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        with torch.no_grad():
            inputs = self.processing_class(
                answer_prompt,
                return_tensors="pt"
            ).to(self.accelerator.device)
            
            # Генерация ответа
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.second_step_generation_config['max_new_tokens'],
                temperature=self.second_step_generation_config['temperature'],
                top_p=self.second_step_generation_config['top_p'],
                do_sample=self.second_step_generation_config['do_sample'],
                pad_token_id=self.second_step_generation_config['pad_token_id'],
                eos_token_id=self.second_step_generation_config['eos_token_id']
            )
            # Декодирование ответа
            output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 
            final_answer = self.processing_class.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

        return final_answer
    
    def _execute_second_step_for_batch(
        self, 
        original_queries: List[str], 
        completion_ids: torch.Tensor,
        solutions: List[str],
        model=None
    ) -> List[float]:
        """
        Выполняет second_step для батча.
        
        Для каждого completion (самоанализа):
        1. Декодируем completion в текст
        2. Извлекаем чистый самоанализ
        3. Генерируем финальный ответ на основе оригинального запроса + самоанализа
        4. Верифицируем финальный ответ
        5. Возвращаем награду
        """
        rewards = []
        
        # Декодируем все completions
        completions_text = self._decode_completions(completion_ids)
        
        for i, (original_query, completion_text) in enumerate(zip(original_queries, completions_text)):
            # Ключ для кэширования
            cache_key = f"{hash(original_query)}_{hash(completion_text)}"
            solution = solutions[i]

            if cache_key in self._second_step_cache:
                reward = self._second_step_cache[cache_key]
            else:
                try:
                    # Извлекаем чистый самоанализ из completion
                   
                    reflection = self._extract_reflection_from_completion(completion_text)
                   
                    # Генерация финального ответа
                    final_answer = self.second_step_func(original_query, reflection, model)
                    #print(final_answer)
                    # Верификация финального ответа
                   
                    reward = reward_function(final_answer, solution, True)
                    #print(reward, solution)
                    # Кэширование результата
                    self._second_step_cache[cache_key] = reward
                   
                    # Обновление статистики
                    self.stats['total_reflections'] += 1
                    if reward > 0.7:
                        self.stats['successful_answers'] += 1
                    else:
                        self.stats['failed_answers'] += 1
                        
                    # Логирование (только для первых нескольких примеров)
                    if i < 2 and self.accelerator.is_main_process:
                        logging.info(
                            f"Second Step Example {i+1} - "
                            f"Original Query: {original_query[:50]}... | "
                            f"Reflection: {reflection[:50]}... | "
                            f"Final Answer: {final_answer[:50]}... | "
                            f"Reward: {reward:.3f}"
                        )
                        
                except Exception as e:
                    logging.error(f"Error in second_step for query {i}: {e}")
                    reward = 0.0
            
            rewards.append(reward)
        
        return rewards
    
    def _calculate_advantages_from_rewards(self, rewards: List[float], num_generations: int) -> torch.Tensor:
        """
        Вычисляет advantages из наград для группы рефлексий.
        Группировка по num_generations (k рефлексий на промпт).
        
        Формула: Advantage_i = (reward_i - mean(rewards)) / std(rewards)
        """
        device = self.accelerator.device
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        k = num_generations
        num_prompts = len(rewards) // k
        
        if num_prompts > 0 and k > 1:
            # Переформатируем в (num_prompts, k)
            grouped_rewards = rewards_tensor.view(num_prompts, k)
            
            # Вычисляем среднее и std для каждой группы
            mean_rewards = grouped_rewards.mean(dim=1, keepdim=True)
            std_rewards = grouped_rewards.std(dim=1, keepdim=True)
            
            # Вычисляем advantages по формуле из статьи
            # Advantage_i = (reward_i - mean(rewards)) / std(rewards)
            advantages = torch.where(
                std_rewards > 0,
                (grouped_rewards - mean_rewards) / (std_rewards + 1e-8),
                grouped_rewards - mean_rewards
            )
            
            # Возвращаем к исходной форме
            advantages = advantages.view(-1)
        else:
            # Если только одна группа или k=1
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std()
            if std_reward > 0:
                advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
            else:
                advantages = rewards_tensor - mean_reward
        
        return advantages
    
    def _split_tensor_dict(self, tensor_dict: Dict[str, Any], num_splits: int) -> List[Dict[str, Any]]:
        """
        Упрощенная версия split_tensor_dict для разделения батча на подбатчи.
        """
        splits = []
        batch_size = None
        
        # Определяем размер батча по первому тензору
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch_size = value.size(0)
                break
        
        if batch_size is None or num_splits <= 0:
            return [tensor_dict]
        
        # Размер каждого подбатча
        split_size = batch_size // num_splits
        if split_size == 0:
            split_size = 1
        
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, batch_size) if i < num_splits - 1 else batch_size
            
            split_dict = {}
            for key, value in tensor_dict.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    # Разделяем тензор по batch dimension
                    split_dict[key] = value[start_idx:end_idx]
                elif isinstance(value, list):
                    # Разделяем список
                    split_dict[key] = value[start_idx:end_idx]
                else:
                    # Копируем остальные значения
                    split_dict[key] = value
            
            splits.append(split_dict)
            
            if end_idx >= batch_size:
                break
        
        return splits
    
    def _shuffle_sequence_dict(self, tensor_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Упрощенная версия shuffle_sequence_dict для перемешивания последовательностей.
        """
        batch_size = None
        
        # Определяем размер батча
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch_size = value.size(0)
                break
        
        if batch_size is None:
            return tensor_dict
        
        # Создаем случайную перестановку
        indices = torch.randperm(batch_size, device=self.accelerator.device)
        
        shuffled_dict = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == batch_size:
                # Перемешиваем тензор
                shuffled_dict[key] = value[indices]
            elif isinstance(value, list) and len(value) == batch_size:
                # Перемешиваем список
                shuffled_list = [value[i] for i in indices.cpu().numpy()]
                shuffled_dict[key] = shuffled_list
            else:
                # Копируем остальные значения
                shuffled_dict[key] = value
        
        return shuffled_dict
    
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        """
        Модифицированный _prepare_inputs с second_step функциональностью.
        
        Реализует трехэтапный процесс R³:
        1. Генерация k самоанализов (рефлексий) на каждый промпт
        2. Для каждого самоанализа: генерация финального ответа и вычисление награды (second_step)
        3. Вычисление advantages и подготовка данных для обновления RL политики
        """
        mode = "train" if self.model.training else "eval"
        
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            
            if self._step % generate_every == 0 or self._buffered_inputs is None or self._first_pass:
                # ============================================================
                # ШАГ 1: ГЕНЕРАЦИЯ САМОАНАЛИЗА (REFLECTIONS)
                # ============================================================
                
                # Вызываем родительский метод для генерации самоанализов
                # generation_batch - это список словарей с промптами
               
                reflection_output = super()._generate_and_score_completions(generation_batch)
            
                # Извлекаем сгенерированные completion_ids (токены самоанализов)
                completion_ids = reflection_output.get('completion_ids')
                if completion_ids is None:
                    raise ValueError("Could not extract completion_ids from reflection_output")
                
                # Извлекаем промпты и ответы из generation_batch
              
                prompts = [item.get('prompt', '') for item in generation_batch]
                solutions = [item.get('solution', '') for item in generation_batch]
                # ============================================================
                # ШАГ 2: ANSWER GENERATION & REWARD CALCULATION (second_step)
                # ============================================================
                
                # Извлекаем оригинальные запросы из промптов
                #print(solutions)
                original_queries = [item.get('original_query', '') for item in generation_batch]
                
                # Важно: num_generations определяет, сколько самоанализов генерируется на каждый промпт
                # В reflection_output completion_ids имеет размер [batch_size * num_generations, seq_len]
                # Нам нужно повторить каждый оригинальный запрос num_generations раз
              #  repeated_original_queries = []
               # for query in original_queries:
                #    repeated_original_queries.extend([query] * self.num_generations)
                
                # Выполняем second_step для каждого самоанализа
                # Для каждого completion (самоанализа) генерируем финальный ответ и вычисляем награду
                second_step_rewards = self._execute_second_step_for_batch(
                    original_queries,
                    completion_ids,
                    solutions, 
                    model=self.model
                )
                
                # ============================================================
                # ШАГ 3: RL POLICY UPDATE PREPARATION
                # ============================================================
                
                # Вычисляем advantages на основе наград от second_step
                advantages = self._calculate_advantages_from_rewards(
                    second_step_rewards, 
                    self.num_generations
                )
                
                # Подменяем advantages в reflection_output
                reflection_output['advantages'] = advantages
                
                # Сохраняем дополнительные данные для отладки и логирования
                reflection_output['second_step_rewards'] = second_step_rewards
                reflection_output['original_queries'] = original_queries
                
                # Разделяем на подбатчи для steps_per_generation
                # Используем упрощенные версии функций
                reflection_output = self._shuffle_sequence_dict(reflection_output)
                generation_batches = self._split_tensor_dict(reflection_output, self.args.steps_per_generation)
                self._buffered_inputs = generation_batches
                
                # Сбрасываем флаг первого прохода
                self._first_pass = False
            
            # Возвращаем подбатч для текущего шага
            current_batch_idx = self._step % self.args.steps_per_generation
            if current_batch_idx < len(self._buffered_inputs):
                inputs = self._buffered_inputs[current_batch_idx]
            else:
                # Если что-то пошло не так, берем первый батч
                inputs = self._buffered_inputs[0] if self._buffered_inputs else {}
            
            self._step += 1
            
            # Логирование статистики
            if mode == "train" and hasattr(self, 'state'):
                if self.state.global_step % 10 == 0 and self.accelerator.is_main_process:
                    if 'second_step_rewards' in inputs:
                        avg_reward = np.mean(inputs['second_step_rewards'])
                        total_reflections = self.stats['total_reflections']
                        successful = self.stats['successful_answers']
                        success_rate = successful / max(total_reflections, 1)
                        
                        logging.info(
                            f"R³ Step {self.state.global_step}: "
                            f"Avg Reward: {avg_reward:.3f}, "
                            f"Success Rate: {success_rate:.1%}, "
                            f"Total Reflections: {total_reflections}, "
                            f"Num Generations (k): {self.num_generations}"
                        )
                        
                        # Сброс статистики каждые 100 шагов
                        if self.state.global_step % 100 == 0:
                            self.stats = {
                                'total_reflections': 0,
                                'successful_answers': 0,
                                'failed_answers': 0
                            }
        else:
            # Для оценки используем стандартный процесс
            inputs = super()._prepare_inputs(generation_batch)
        
        return inputs

def custom_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs
) -> list[float | None]:
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, sol in zip(contents, solution, strict=True):
        rewards.append(reward_function(content, sol, True))
    return rewards
# Пример создания task-specific верификатора для математических задач
def create_verifier() -> Callable:
    """
    Создает верификатор для наших задач.
    """

    return custom_reward

# Пример использования
if __name__ == "__main__":
    from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig
    
    # Инициализация
    file_path = sys.argv[1]
    model_name = sys.argv[2]

    dataset = load_dataset("json", data_files = file_path, split="train")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)

    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=2000,
        max_completion_length=2000,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        use_vllm=False,
        vllm_mode="colocate",
        log_on_each_node=False,
        vllm_gpu_memory_utilization=.3,
        output_dir="outputs/Qwen3-1.7B_R3",
        report_to="none" 
    )

    
    # Создаем R³ тренер
    trainer = R3GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class = tokenizer,
        reward_funcs=custom_reward
    )
    
    # Запуск тренировки
    trainer.train()
