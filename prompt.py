special_token_prompt= {
    "support_judge":(
        "You will receive an instruction, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement\n"
        "Use the following entailment scale to generate a score:\n"
        "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.\n"
        "[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].\n"
        "[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
        "Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
    ),
    "support_input":(
        "Task instruction: {instruction}\n"
        "Evidence: {evidence}\n"
        "Output: {output}"
    ),
    "relevant_judge":(
        "When given instruction and evidence, evaluate whether the evidence is relevant to the instruction and provides valuable information for generating meaningful responses.\n"
        "Use a rating of [Relevant] to indicate relevance and usefulness, and [Irrelevant] to indicate irrelevance."
    ),
    "relevant_input": (
        "Task instruction: {instruction}\n"
        "Evidence: {evidence}"
    ),
    "useful_judge": (
        "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n"
        "[Utility:5]: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
        "[Utility:4]: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
        "[Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
        "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
        "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
    ),
    "useful_input": (
        "Task instruction: {instruction}\n"
        "Output: {output}"
    ),
}
generate_answer_prompt={
    "generate_answer": (
        "You will be given a task instruction and an evidence,your task is to provide an answer that appropriately satisfy the request.\n"
        "You should refer to the evidence to provide the answer.\n\n"
    ),
    "generate_answer_input":(
        "task instruction: {instruction}\n"
        "evidence: {evidence}"
    ),
    "generate_answer_with_critic":(
        "You will be given a task instruction,evidence,preceding answer and the criticism to the preceding answer,your task is to provide an better answer based on the critic to the preceding answer.\n"

    ),
    "generate_answer_with_critic_input":(
        "task instruction: {instruction}\n"
        "evidence: {evidence}\n"
        "preceding answer:{preceding answer}\n"
    ),
}
critic_prompt={
    "critic_useful":(
        "You will be given a task instruction,output and a score that rate how helpful an informative the output is to the query,from 1 (lowest) - 5 (highest). We call this score perceived utility.\n"
        "The scores are divided into five levels:\n"
        "[Utility:5]: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
        "[Utility:4]: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
        "[Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
        "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
        "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
        "Your task is to criticize the output from the perspectives of helpfulness and informativeness."
        "You should generate the answer in the following fotmat:\n"
        "criticism:your answer."
    ),
    "critic_useful_input": (
        "task instruction: {instruction}\n"
        "output:{output}\n"
        "score:{score}\n"
    ),

    "critic_support":(
        "You will be given a task instruction,evidence,output and a score that rate how much information in the output is entailed by the evidence."
        "The scores are divided into three levels:\n"
        "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.\n"
        "[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].\n"
        "[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
        "Your task is to criticize the output based on the score.\n"
        "You should generate the answer in the following fotmat:\n"
        "criticism:your answer."
    ),
    "critic_support_input": (
        "task instruction: {instruction}\n"
        "evidence: {evidence}\n"
        "output:{output}\n"
        "score:{score}\n"
    ),

}

task_type_prompt = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
            "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
            "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
            "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
            "arc": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
            "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
            "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}