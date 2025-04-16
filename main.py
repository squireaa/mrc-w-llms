from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import os
import time
import logging
import json
from typing import List
from data_generators import gen_reclor, gen_logicqa
import nltk
from nltk.tokenize import sent_tokenize

def batch_inference(model, tokenizer, device, dialogs: List[List[dict]]) -> List[str]:
    responses = []
    for i, dialog in enumerate(dialogs):
        print(f"{i}")
        encodeds = tokenizer.apply_chat_template(dialog, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        responses.append(decoded[0])
    return responses

def prepare_dialogs(context, question, choices):
    message_pairs = []
    prompt = [
        {
            "role": "system",
            "content": "You are an advanced language model specialized in logical reasoning and machine reading comprehension tasks. Your goal is to carefully evaluate the context and the question, then select the **most logical and evidence-based answer** from the given choices. Provide a step-by-step justification that clearly explains your reasoning."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}\n\nChoices: {choices}\n\nStep-by-step:\n1. Identify the cause (A) and the effect (B) in the context.\n2. Analyze whether A normally leads to B and check if there are conditions that might disrupt this causation.\n3. Match the reasoning to the provided choices.\n4. Select the best answer and provide a clear, evidence-based explanation."
        }
    ]
    message_pairs.append(prompt)
    return message_pairs
def isolate_answer(text):
    last_inst_index = text.rfind("[/INST]")
    if last_inst_index != -1:
        return text[last_inst_index + len("[/INST]"):]
def main():
    # Set up the Mistral v0.3 model
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda"
    # cache_dir = "/home/as9df/.cache/huggingface/hub/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3").to(device)
    data = gen_logicqa()
    samples = 5
    prompts = []
    answers = []
    for n in range(0, samples):
        context = data[n]['context']
        question = data[n]['question']
        choices = data[n]['choices']
        answer = data[n]['answer']
        prompts += prepare_dialogs(context, question, choices)
        answers += [answer]
    responses = batch_inference(model, tokenizer, device, prompts)
    output_file = f"example_output.txt"
    with open(output_file, 'w') as file:
        for i, response in enumerate(responses):
            file.write("machine answer: " + isolate_answer(response) + "\n")
            file.write("actual answer: " + answers[i] + "\n" + "==========================================================" + "\n\n")

if __name__ == "__main__":
    a = time.time()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    main()
    b = time.time()
    print(f"Finished in {b - a} seconds.")