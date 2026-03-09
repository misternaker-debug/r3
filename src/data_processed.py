from datasets import Dataset
import sys
from transformers import AutoTokenizer
   
def process_data(dataset_path, model_name):
    dataset = Dataset.from_json(dataset_path)
    system_prompt = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dataset_path[-14:] == 'train_kfu.json':
        new_dataset = {}
        new_dataset['answer'] = []
        new_dataset['question'] = []
        for data in dataset['messages']:
            new_dataset['question'].append(data[0]['content'])
            new_dataset['answer'].append(data[1]['content'])
        dataset = Dataset.from_dict(new_dataset)

    dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"][0]['content']},
    ],
    "solution": x["ground_truth"],
    })
    date_formatted = dataset.map(
         lambda x: {"formatted" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = False)},
         batched = True,
    ) 
    print('format data:', date_formatted[0])
    return date_formatted

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        model_name = sys.argv[2]
        process_data(file_path, model_name)
