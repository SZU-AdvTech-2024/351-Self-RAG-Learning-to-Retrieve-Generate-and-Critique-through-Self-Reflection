from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig
from passage_retrieval import Retriever
import argparse
import src.slurm
from process_data import load_data
from process_data import load_prompt_critic
from process_data import load_prompt_generate
from process_data import load_prompt_special_token
import re


def generate_result(args):
    src.slurm.init_distributed_mode(args)

    retriever = Retriever(args)
    retriever.setup_retriever()

    critic_model_path = args.critic_model_path
    generate_model_path=args.generate_model_path

    critic_tokenizer = AutoTokenizer.from_pretrained(critic_model_path)
    critic_model = AutoModelForCausalLM.from_pretrained(critic_model_path, device_map="auto")

    generate_tokenizer=AutoTokenizer.from_pretrained(generate_model_path)
    generate_model= AutoModelForCausalLM.from_pretrained(generate_model_path, device_map="auto")

    input_data = load_data(args.input_file)

    for item in input_data:
        if args.dataset_name == "arc":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            item["question"] = item["question"] + choices
            item["ground_answer"] = item["answerKey"]

        if args.dataset_name == "fever":
            item["question"]=item["claim"]
            item["ground_answer"]=item["answer"][0]

        question = item["question"]  # 提取数据中的问题
        retrieve_context_count = 0
        retrieve_context_list = []
        if args.retrieve_mode != "with_retrieve_context":
            retrieve_context_list = retriever.search_document(question, args.n_docs)
            retrieve_context = retrieve_context_list[retrieve_context_count]  # 使用检索函数
            item["evidence"] = retrieve_context
            retrieve_context_count += 1

        else:
            retrieve_context_dict_list = item["ctxs"]
            for i in range(len(retrieve_context_dict_list)):
                retrieve_context_list.append(retrieve_context_dict_list[i]["text"])
            retrieve_context = retrieve_context_list[retrieve_context_count]  # 提取数据中的检索文档
            item["evidence"] = retrieve_context
            retrieve_context_count += 1

        relevance_prompt_input = load_prompt_special_token(item, "relevant", args.dataset_name)
        relevance_prompt = critic_tokenizer(relevance_prompt_input, return_tensors="pt").to("cuda")

        generate_special_token_config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.6,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.2,
        )

        outputs = critic_model.generate(**relevance_prompt, generation_config=generate_special_token_config)
        relevance_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)
        relevance_token_score = re.findall(r"\[(.*?)\]", relevance_token)[0]

        while relevance_token_score != "Relevant":
            if retrieve_context_count < len(retrieve_context_list):
                retrieve_context = retrieve_context_list[retrieve_context_count]  # 使用检索函数
                item["evidence"] = retrieve_context

                relevance_prompt_input = load_prompt_special_token(item, "relevant", args.dataset_name)
                relevance_prompt = critic_tokenizer(relevance_prompt_input, return_tensors="pt").to("cuda")
                outputs = critic_model.generate(**relevance_prompt, generation_config=generate_special_token_config)
                relevance_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)
                relevance_token_score = re.findall(r"\[(.*?)\]", relevance_token)[0]

        generate_answer_input = load_prompt_generate(item, "without_critic", args.dataset_name)
        generate_answer_prompt = generate_tokenizer(generate_answer_input, return_tensors="pt").to("cuda")

        generate_answer_config = GenerationConfig(
            max_new_tokens=500,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
        )

        outputs = generate_model.generate(**generate_answer_prompt, generation_config=generate_answer_config)

        answer = generate_tokenizer.decode(outputs[0], skip_special_tokens=True)
        item["predict_answer"] = answer

        support_prompt_input = load_prompt_special_token(item, "support", args.dataset_name)
        support_prompt = critic_tokenizer(support_prompt_input, return_tensors="pt").to("cuda")

        outputs = critic_model.generate(**support_prompt, generation_config=generate_special_token_config)
        support_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

        useful_prompt_input = load_prompt_special_token(item, "useful", args.dataset_name)
        useful_prompt = critic_tokenizer(useful_prompt_input, return_tensors="pt").to("cuda")

        outputs = critic_model.generate(**useful_prompt, generation_config=generate_special_token_config)
        useful_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

        support_token_score = re.findall(r"\[.*?\]", support_token)[0]  # 从support_token中提取score
        useful_token_score = re.findall(r"\[.*?\]", useful_token)[0]  # 从useful_token中提取score

        support_critic_count = 0
        critic_count = 0
        while (support_token_score == "[No support / Contradictory]") or (
                support_token_score == "[Partially supported]" and useful_token_score[-2] < "5") or (
                support_token_score == "[Fully supported]" and useful_token_score[-2] < "4"):

            support_critic = None
            useful_critic = None

            if support_token_score == "[No support / Contradictory]":
                if support_critic_count == 3:
                    if retrieve_context_count < len(retrieve_context_list):
                        retrieve_context = retrieve_context_list[retrieve_context_count]  # 使用检索函数
                        item["evidence"] = retrieve_context
                        retrieve_context_count += 1

                        relevance_prompt_input = load_prompt_special_token(item, "relevant", args.dataset_name)
                        relevance_prompt = critic_tokenizer(relevance_prompt_input, return_tensors="pt").to("cuda")
                        outputs = critic_model.generate(**relevance_prompt, generation_config=generate_special_token_config)
                        relevance_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        relevance_token_score = re.findall(r"\[(.*?)\]", relevance_token)[0]

                    while relevance_token_score != "Relevant":
                        if retrieve_context_count < len(retrieve_context_list):
                            retrieve_context = retrieve_context_list[retrieve_context_count]  # 使用检索函数
                            item["evidence"] = retrieve_context
                            retrieve_context_count += 1

                            relevance_prompt_input = load_prompt_special_token(item, "relevant", args.dataset_name)
                            relevance_prompt = critic_tokenizer(relevance_prompt_input, return_tensors="pt").to("cuda")
                            outputs = critic_model.generate(**relevance_prompt, generation_config=generate_special_token_config)
                            relevance_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)
                            relevance_token_score = re.findall(r"\[(.*?)\]", relevance_token)[0]

                    generate_answer_input = load_prompt_generate(item, "without_critic", args.dataset_name)
                    generate_answer_prompt = generate_tokenizer(generate_answer_input, return_tensors="pt").to("cuda")

                    outputs = generate_model.generate(**generate_answer_prompt, generation_config=generate_answer_config)
                    answer = generate_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    item["predict_answer"]=answer

                    support_prompt_input = load_prompt_special_token(item, "support", args.dataset_name)
                    support_prompt = critic_tokenizer(support_prompt_input, return_tensors="pt").to("cuda")

                    outputs = critic_model.generate(**support_prompt, generation_config=generate_special_token_config)
                    support_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

                    useful_prompt_input = load_prompt_special_token(item, "useful", args.dataset_name)
                    useful_prompt = critic_tokenizer(useful_prompt_input, return_tensors="pt").to("cuda")

                    outputs = critic_model.generate(**useful_prompt, generation_config=generate_special_token_config)
                    useful_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

                    support_token_score = re.findall(r"\[.*?\]", support_token)[0]  # 从support_token中提取score
                    useful_token_score = re.findall(r"\[.*?\]", useful_token)[0]  # 从useful_token中提取score

                    support_critic_count = 0
                    continue

                support_critic = load_prompt_critic(item, "critic_support",support_token_score, args.dataset_name)  # 调用critic函数
                support_critic_count += 1

            if useful_token_score < "5":
                useful_critic = load_prompt_critic(item, "critic_useful", useful_token_score,args.dataset_name)

            generate_with_critic_prompt_input = load_prompt_generate(item, "with_critic", args.dataset_name)

            if support_critic != None and useful_critic != None:
                generate_with_critic_prompt_input = generate_with_critic_prompt_input + "criticism:" + support_critic.replace(
                    "criticism:", "") + useful_critic.replace("criticism:", "")
            if support_critic == None and useful_critic != None:
                generate_with_critic_prompt_input += useful_critic.replace("criticism:", "")

            if support_critic != None and useful_critic == None:
                generate_with_critic_prompt_input += support_critic.replace("criticism:", "")

            generate_with_critic_prompt = generate_tokenizer(generate_with_critic_prompt_input, return_tensors="pt").to("cuda")
            outputs = generate_model.generate(**generate_with_critic_prompt, generation_config=generate_answer_config)
            answer = generate_tokenizer.decode(outputs[0], skip_special_tokens=True)
            item["predict_answer"] = answer

            support_prompt_input = load_prompt_special_token(item, "support", args.dataset_name)
            support_prompt = critic_tokenizer(support_prompt_input, return_tensors="pt").to("cuda")

            outputs = critic_model.generate(**support_prompt, generation_config=generate_special_token_config)
            support_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

            useful_prompt_input = load_prompt_special_token(item, "useful", args.dataset_name)
            useful_prompt = critic_tokenizer(useful_prompt_input, return_tensors="pt").to("cuda")

            outputs = critic_model.generate(**useful_prompt, generation_config=generate_special_token_config)
            useful_token = critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

            support_token_score = re.findall(r"\[.*?\]", support_token)[0]  # 从support_token中提取score
            useful_token_score = re.findall(r"\[.*?\]", useful_token)[0]  # 从useful_token中提取score

            critic_count += 1
            if critic_count == args.max_critic:
                break

    return input_data

def arc_and_fever_evaluate(data):
    sum_grade=0
    for item in data:
        if item["predict_answer"] == item["ground_answer"]:
            sum_grade+=100
            print("The result is true!")
        else:
            print("The result is false!")

    average_accuracy=sum_grade/len(data)
    return average_accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--critic_model_path", type=str, help="path to critic model")
    parser.add_argument("--generate_model_path", type=str, help="path to generate model")
    parser.add_argument("--retrieve_model_path", type=str, help="path to retrieve model")
    parser.add_argument("--retrieve_mode", type=str, help="with_retrieve_context or without_retrieve_context")
    parser.add_argument("--input_file", type=str, help="test data")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--max_critic", type=str, help="max_critic_times")
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="path passages_embeddings")
    parser.add_argument("--n_docs", type=int, default=20, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    args = parser.parse_args()

    data_processed = generate_result(args)

    if args.dataset_name =="arc" or args.dataset_name =="fever":
        average_accuracy=arc_and_fever_evaluate(data_processed)
        print("The average accuracy is {}%".format(average_accuracy))