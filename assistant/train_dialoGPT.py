from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load the tokenizer and model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token as the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset and split into train and eval
dataset = load_dataset("json", data_files="dataset.json")
dataset = dataset["train"].train_test_split(test_size=0.2)  # 80% train, 20% eval


# Define the tokenization function
def tokenize_function(examples):
    inputs = [
        inp + tokenizer.eos_token + resp + tokenizer.eos_token
        for inp, resp in zip(examples["input"], examples["response"])
    ]
    model_inputs = tokenizer(
        inputs, truncation=True, padding="max_length", max_length=128
    )

    # Use input_ids as labels for language modeling tasks
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
)

# Set up the Trainer with eval_dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Fine-tune the model
trainer.train()
# Save the fine-tuned model and tokenizer
trainer.save_model("fine_tuned_dialoGPT")
tokenizer.save_pretrained("fine_tuned_dialoGPT")
