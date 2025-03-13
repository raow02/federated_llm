import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        Custom dataset for fine-tuning. Each sample in the JSON data contains:
        "instruction", "context", "response", "category".
        The training text is formed by concatenating these fields.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Concatenate instruction, context, and response to form the training text.
        text = f"Instruction: {sample.get('instruction', '')}\nContext: {sample.get('context', '')}\nResponse: {sample.get('response', '')}"
        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        # Remove batch dimension.
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

def main():
    parser = argparse.ArgumentParser(description="Federated LLM fine-tuning with LoRA on specified domain-specific datasets.")
    parser.add_argument("--base_model", type=str, default="NousResearch/Llama-2-7b-hf",
                        help="Name or path of the base Llama 7B model.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing training JSON files named as local_training_{domain}.json.")
    parser.add_argument("--domain", type=str, nargs='+', required=True,
                        help="List of domains to train on. For each domain, a file named local_training_{domain}.json must exist.")
    parser.add_argument("--output_dir", type=str, default="output_models",
                        help="Directory to save the fine-tuned LoRA models and baseline model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per domain.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    args = parser.parse_args()

    # Create output directory if not exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Define LoRA configuration.
    lora_config = LoraConfig(
        r=8,               # Rank of the LoRA update matrices.
        lora_alpha=16,     # Scaling factor.
        target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA.
        lora_dropout=0.1,  # Dropout probability for LoRA layers.
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Process each specified domain.
    for domain in args.domain:
        file_path = os.path.join(args.data_dir, f"local_training_{domain}.json")
        if not os.path.exists(file_path):
            print(f"Training file for domain '{domain}' not found at path: {file_path}")
            continue

        print(f"\nStarting fine-tuning for domain: {domain}")

        # Load domain-specific dataset.
        with open(file_path, 'r') as f:
            data = json.load(f)
        dataset = CustomDataset(data, tokenizer)

        # Load a fresh copy of the base model and apply LoRA.
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", load_in_8bit=True)
        model = get_peft_model(model, lora_config)

        # Define training arguments.
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"{domain}_lora"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            fp16=True,
            optim="adamw_torch",
            report_to="none"  # Disable reporting to third-party services.
        )

        # Initialize the Trainer.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        # Fine-tune the model.
        trainer.train()

        # Save the LoRA parameters for this domain.
        domain_output_dir = os.path.join(args.output_dir, f"{domain}_lora")
        model.save_pretrained(domain_output_dir)
        print(f"Saved LoRA parameters for domain '{domain}' at {domain_output_dir}")

if __name__ == "__main__":
    main()
