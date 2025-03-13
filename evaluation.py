import argparse
import os
import json
import re
import random
import time
import pandas as pd

import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Define the GPT-4 evaluation function with an unbiased prompt
def evaluate_with_gpt4(instruction, context, response_a, response_b, client):
    """
    Use GPT-4 to compare two model responses and return a score pair.
    
    Parameters:
    - instruction: The original instruction text.
    - context: Additional context for the instruction.
    - response_a: The response from Model A.
    - response_b: The response from Model B.
    - client: The OpenAI client instance.
    
    Returns:
    - The text result returned by GPT-4 in the following format:
      - Model A Score: X
      - Model B Score: Y
      - Better Model: [A / B / Tie]
    """
    # Build prompt based on whether context is provided
    if context:
        instruction_text = f"""Instruction:
"{instruction}"

Context:
"{context}" """
    else:
        instruction_text = f"""Instruction:
"{instruction}" """
        
    prompt = f"""
You are an expert language model evaluator. Given an instruction and context, compare two model responses.
Score each response on a scale from 1 to 10 (10 being the best) based on:
- Relevance to the instruction and context
- Coherence and fluency
- Completeness of the answer
- Factual correctness

{instruction_text}

Response from **Model A**:
"{response_a}"

Response from **Model B**:
"{response_b}"

Now, rate each response from 1 to 10 and state which one is better. Return in this format:

- Model A Score: X
- Model B Score: Y
- Better Model: [A / B / Tie]
    """
    try:
        # Call GPT-4 API using the new client format
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Set temperature to 0 to reduce randomness
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error during GPT-4 evaluation:", e)
        return None

def parse_scores(gpt4_result):
    """
    Parse the score pair from the GPT-4 returned result.
    Returns (model_a_score, model_b_score)
    """
    try:
        model_a_score = int(re.search(r"Model A Score: (\d+)", gpt4_result).group(1))
        model_b_score = int(re.search(r"Model B Score: (\d+)", gpt4_result).group(1))
        return model_a_score, model_b_score
    except Exception as e:
        print("Error parsing GPT-4 result:", e)
        return None, None

def generate_response(model, tokenizer, instruction, context="", max_length=512, device="cuda"):
    """
    Generate a response using the specified model and tokenizer.
    
    Parameters:
    - model: The language model to use for generation
    - tokenizer: The tokenizer for the model
    - instruction: The instruction for the model
    - context: Optional context to provide with the instruction
    - max_length: Maximum length of the generated response
    - device: Device to run generation on (cuda or cpu)
    
    Returns:
    - The generated response text
    """
    # Combine instruction and context if context is provided
    if context:
        input_text = f"Instruction: {instruction}\nContext: {context}"
    else:
        input_text = instruction
        
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned models vs. Original model using GPT-4 auto-evaluation.")
    parser.add_argument("--base_model", type=str, default="NousResearch/Llama-2-7b-hf",
                        help="Name or path of the base model.")
    parser.add_argument("--output_dir", type=str, default="output_models",
                        help="Directory where fine-tuned LoRA models are saved (folder: {domain}_lora).")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the global test JSON file (global_test.json).")
    parser.add_argument("--domain", type=str, nargs='+', default=["brainstorming", "classification", "closed_qa", "creative_writing", "general_qa", "information_extraction", "open_qa", "summarization"],
                        help="List of domain names corresponding to the fine-tuned models and test sample categories.")
    parser.add_argument("--n_eval_repeat", type=int, default=1,
                        help="Number of times to repeat GPT-4 evaluation per sample to average out randomness.")
    parser.add_argument("--max_test_samples", type=int, default=10,
                        help="For quick testing, you can limit the number of test samples per domain (set to -1 to use all).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run generation on (e.g., cuda or cpu).")
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenAI API key to access GPT-4 evaluation.")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory to save evaluation results. Default is current working directory.")
    parser.add_argument("--randomize_order", type=bool, default=True,
                        help="Randomly assign fine-tuned and original models as A or B to reduce bias.")
    args = parser.parse_args()

    # Create the save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Set up OpenAI client with the new API format
    client = openai.OpenAI(api_key=args.api_key)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load original model (used to generate the original responses)
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        load_in_8bit=True,
    )
    original_model.eval()

    # Load global test data
    global_test_path = os.path.join(args.data_dir, "global_test.json")
    if not os.path.exists(global_test_path):
        print(f"Global test file not found at {global_test_path}")
        return
    with open(global_test_path, "r", encoding="utf-8") as f:
        global_test_data = json.load(f)

    # Group the test data by category (each category corresponds to a domain)
    test_data_by_domain = {domain: [] for domain in args.domain}
    for sample in global_test_data:
        category = sample.get("category", "").strip()
        if category in test_data_by_domain:
            test_data_by_domain[category].append(sample)
    print("Test data distribution by domain:")
    for domain, samples in test_data_by_domain.items():
        print(f"  {domain}: {len(samples)} samples")

    # Save the evaluation results in a dictionary, structured as {finetuned_domain: {test_domain: (avg_ft_score, avg_orig_score)}}.
    eval_results = {ft_domain: {} for ft_domain in args.domain}

    # For each fine-tuned model (obtained by training on each domain)
    for ft_domain in args.domain:
        model_path = os.path.join(args.output_dir, f"{ft_domain}_lora")
        if not os.path.exists(model_path):
            print(f"Fine-tuned model for domain '{ft_domain}' not found at {model_path}")
            continue
        print(f"\nLoading fine-tuned model for domain: {ft_domain}")
        base_model_ft = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            load_in_8bit=True,
        )
        ft_model = PeftModel.from_pretrained(base_model_ft, model_path)
        ft_model.eval()

        # For each test domain
        for test_domain in args.domain:
            samples = test_data_by_domain.get(test_domain, [])
            if not samples:
                print(f"No test samples found for domain: {test_domain}")
                continue

            print(f"\nEvaluating Fine-Tuned Model '{ft_domain}' on test domain '{test_domain}' with {len(samples)} samples...")
            ft_scores = []
            orig_scores = []
            
            # If max_test_samples is set, test with a subset of samples
            if args.max_test_samples > 0:
                samples = samples[:args.max_test_samples]
            
            # For each test sample
            for sample in samples:
                instruction = sample.get("instruction", "").strip()
                # Extract both instruction and context
                instruction = sample.get("instruction", "").strip()
                context = sample.get("context", "").strip()
                
                # Generate responses for both models
                response_ft = generate_response(ft_model, tokenizer, instruction, context, device=args.device)
                response_orig = generate_response(original_model, tokenizer, instruction, context, device=args.device)
                
                # Repeat the evaluation n_eval_repeat times for each sample
                sample_ft_scores = []
                sample_orig_scores = []
                for i in range(args.n_eval_repeat):
                    print(f"Evaluating sample for domain '{test_domain}' (repeat {i+1}/{args.n_eval_repeat})...")
                    
                    # Randomize which model is A and which is B to reduce bias
                    if args.randomize_order and random.random() > 0.5:
                        # Fine-tuned model is A, original model is B
                        response_a, response_b = response_ft, response_orig
                        is_ft_model_a = True
                    else:
                        # Original model is A, fine-tuned model is B
                        response_a, response_b = response_orig, response_ft
                        is_ft_model_a = False
                        
                    gpt4_result = evaluate_with_gpt4(instruction, context, response_a, response_b, client)
                    # If the call fails, wait a few seconds and retry
                    retry_count = 0
                    while gpt4_result is None and retry_count < 3:
                        time.sleep(5)
                        gpt4_result = evaluate_with_gpt4(instruction, context, response_a, response_b, client)
                        retry_count += 1
                    if gpt4_result is None:
                        print("Skipping this evaluation due to repeated errors.")
                        continue
                        
                    model_a_score, model_b_score = parse_scores(gpt4_result)
                    if model_a_score is not None and model_b_score is not None:
                        # Map the scores back to ft_model and orig_model based on which was A and which was B
                        if is_ft_model_a:
                            ft_score, orig_score = model_a_score, model_b_score
                        else:
                            ft_score, orig_score = model_b_score, model_a_score
                        
                        sample_ft_scores.append(ft_score)
                        sample_orig_scores.append(orig_score)
                        
                    # To avoid frequent requests, wait a bit
                    time.sleep(2)
                    
                # If there are successful evaluations for this sample, take the average
                if sample_ft_scores and sample_orig_scores:
                    avg_ft = sum(sample_ft_scores) / len(sample_ft_scores)
                    avg_orig = sum(sample_orig_scores) / len(sample_orig_scores)
                    ft_scores.append(avg_ft)
                    orig_scores.append(avg_orig)
                    
            # Compute the overall average score for the (ft_domain, test_domain) combination
            if ft_scores and orig_scores:
                overall_ft = sum(ft_scores) / len(ft_scores)
                overall_orig = sum(orig_scores) / len(orig_scores)
                eval_results[ft_domain][test_domain] = (overall_ft, overall_orig)
                print(f"Result for Fine-Tuned '{ft_domain}' on test domain '{test_domain}': (Fine-Tuned Score: {overall_ft:.2f}, Original Score: {overall_orig:.2f})")
            else:
                eval_results[ft_domain][test_domain] = (None, None)
                print(f"No valid evaluations for Fine-Tuned '{ft_domain}' on test domain '{test_domain}'.")

    # Save the evaluation results as JSON
    output_file = os.path.join(args.save_dir, "evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"\nEvaluation complete! Results saved to {output_file}")

    # Convert the nested dictionary into a DataFrame for better visualization
    df = pd.DataFrame.from_dict(eval_results, orient="index")

    # Save DataFrame as CSV
    csv_file = os.path.join(args.save_dir, "evaluation_results.csv")
    df.to_csv(csv_file)
    print(f"Results also saved as CSV to {csv_file}")

    # Print the table
    print("\nEvaluation Results Table:")
    print(df)

if __name__ == "__main__":
    main()