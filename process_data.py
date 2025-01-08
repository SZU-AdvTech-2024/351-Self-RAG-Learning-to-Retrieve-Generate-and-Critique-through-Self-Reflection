from prompt import special_token_prompt
from prompt import generate_answer_prompt
from prompt import critic_prompt
from prompt import task_type_prompt
import jsonlines
import json
task_type=["wow", "fever", "arc_easy", "arc_hard", "obqa", "qrecc", "race", "asqa"]


def load_data(input_file_path):
    if input_file_path.endswith(".json"):
        with open(input_file_path, 'r') as json_f:
            input_data = json.load(json_f)
    else:
        with jsonlines.open(input_file_path, 'r') as jsonl_f:
            input_data = [item for item in jsonl_f]

    return  input_data

def load_prompt_special_token(item,mode,dataset_name):
    if mode=="support" :
        if dataset_name in task_type:
            instruction = task_type_prompt[dataset_name]+"## input:\n"+item["question"]
        else:
            instruction = item["question"]
        evidence = item["evidence"]
        output = item["predict_answer"]
        dict={"instruction":instruction,"evidence":evidence,"output":output}

        input =special_token_prompt["support_input"].format_map(dict)

        prompt_input=(
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
        prompt_dict={"instruction":special_token_prompt["support_judge"],"input":input}

        prompt=prompt_input.format_map(prompt_dict)
    elif mode=="relevant":
        if dataset_name in task_type:
            instruction = task_type_prompt[dataset_name]+"## input:\n"+item["question"]
        else:
            instruction = item["question"]
        evidence = item["evidence"]
        dict = {"instruction": instruction, "evidence": evidence}

        input = special_token_prompt["relevant_input"].format_map(dict)

        prompt_input = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
        prompt_dict = {"instruction": special_token_prompt["relevant_judge"], "input": input}

        prompt = prompt_input.format_map(prompt_dict)
    else:
        if dataset_name in task_type:
            instruction = task_type_prompt[dataset_name]+"## input:\n"+item["question"]
        else:
            instruction = item["question"]
        output = item["predict_answer"]
        dict={"instruction":instruction,"output":output}

        input = special_token_prompt["useful_input"].format_map(dict)

        prompt_input = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
        prompt_dict = {"instruction": special_token_prompt["useful_judge"], "input": input}
        prompt=prompt_input.format_map(prompt_dict)
    return  prompt
def load_prompt_generate(item,mode,dataset_name):
    if mode=="with_critic" :
      if dataset_name in task_type:
        instruction = task_type_prompt[dataset_name]+"## input:\n"+item["question"]
      else:
        instruction = item["question"]
      evidence = item["evidence"]
      answer=item["predict_answer"]
      dict={"instruction":instruction,"evidence":evidence,"preceding answer":answer}

      input = generate_answer_prompt["generate_answer_with_critic_input"].format_map(dict)
      prompt = generate_answer_prompt["generate_answer_with_critic"] + input

    else:
      if dataset_name in task_type:
        instruction = task_type_prompt[dataset_name]+"## input:\n"+item["question"]
      else:
        instruction = item["question"]
      evidence = item["evidence"]
      dict = {"instruction": instruction, "evidence": evidence}

      input=generate_answer_prompt["generate_answer_input"].format_map(dict)
      prompt=generate_answer_prompt["generate_answer"] +input

    return prompt
def load_prompt_critic(item,mode,score,dataset_name):
    if mode=="critic_useful" :
        if dataset_name in task_type:
            instruction = task_type_prompt[dataset_name]+"## input:\n"+item["question"]
        else:
            instruction = item["question"]
        output = item["predict_answer"]
        dict = {"instruction": instruction, "output": output,"score":score}

        input = critic_prompt["critic_useful_input"].format_map(dict)
        prompt = critic_prompt["critic_useful"] + input

    else:
        if dataset_name in task_type:
            instruction = task_type_prompt[dataset_name]+"## input:\n"+item["question"]
        else:
            instruction = item["question"]
        evidence = item["evidence"]
        output = item["predict_answer"]
        dict = {"instruction": instruction, "evidence": evidence, "output": output,"score":score}

        input=critic_prompt["critic_support_input"].format_map(dict)
        prompt=critic_prompt["critic_support"] + input

    return prompt