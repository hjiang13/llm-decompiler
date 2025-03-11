import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# 1. Load the Dataset (JSONL format)
dataset = load_dataset("json", data_files={"train": "../benchmarks/dataset.jsonl"}, split="train")

# 2. Specify the local model path (update this with your actual local path)
local_model_path = "/path/to/local/LLMCompiler7B"  # e.g., "/data2/models/LLMCompiler7B"

# 3. Load the tokenizer and model from your local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)

# 4. Define a tokenization function for both the input (IR) and target (C++ code)
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input"], truncation=True, max_length=4096)
    # Use the tokenizer as a target tokenizer for the labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], truncation=True, max_length=4096)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 5. Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 6. Define training arguments
training_args = TrainingArguments(
    output_dir="./llm_decompiler_output",
    num_train_epochs=3,                # Adjust epochs as needed
    per_device_train_batch_size=2,     # Adjust based on your GPU memory
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="no",          # Update if you add a validation set
)

# 7. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 8. Start Training
trainer.train()
