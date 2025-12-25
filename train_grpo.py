import re
import sys 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import GRPOConfig, GRPOTrainer
from sklearn.metrics.pairwise import cosine_similarity
import data_procesed

def train_grpo(dataset, model):
    
    training_args = GRPOConfig(
            temperature = 1.0,
            learning_rate = 5e-6,
            weight_decay = 0.001,
            warmup_ratio = 0.1,
            lr_scheduler_type = "linear",
            logging_steps = 1,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4, 
            num_generations = 4, 
            report_to = "none", 
            output_dir = "outputs",
            )
    trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = [
                match_format_approximately,
                match_format_exactly,
                semantic_similarity
            ],
            args = training_args,
            train_dataset = dataset,
            )
    trainer.train()

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores


model_judge = SentenceTransformer('all-MiniLM-L6-v2')

def match_format_approximately(prompts, completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)

    return scores

def semantic_similarity(prompts, completions, answer, **kwargs):
    """Семантическое сходство через косинусное расстояние эмбеддингов"""
    scores = []
    treshold = 0.7
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        embedding_res = model_judge.encode(response)
        embedding_ans = model_judge.encode(answer[0])
        score = 1 if cosine_similarity([embedding_res], [embedding_ans]) >= treshold else 0
        scores.append(score)
  
    return scores

def match_format_exactly(prompts, completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        model_name = sys.argv[2]
        dataset = data_procesed.process_data(file_path, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                )
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        if hasattr(model, 'generation_config') and model.generation_config is not None:
            model.generation_config.bos_token_id = tokenizer.bos_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        reasoning_start = "<start_working_out>" # Acts as <think>
        reasoning_end   = "<end_working_out>"   # Acts as </think>
        solution_start  = "<SOLUTION>"
        solution_end    = "</SOLUTION>"

        system_prompt = \
            f"""You are given a problem.
            Think about the problem and provide your working out.
            Place it between {reasoning_start} and {reasoning_end}.
            Then, provide your solution between {solution_start}{solution_end}"""

        solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
            "(?:" + re.escape(tokenizer.eos_token) + ")?"

        match_format = re.compile(
            rf"{reasoning_end}.*?"\
            rf"{solution_start}(.+?){solution_end_regex}"\
            rf"[\s]{{0,}}$",
            flags = re.MULTILINE | re.DOTALL
            )
        train_grpo(dataset, model)
