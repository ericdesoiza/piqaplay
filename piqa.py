from datasets import load_dataset
from transformers import GPT2Tokenizer

# 1. Load the PIQA dataset
piqa_dataset = load_dataset("piqa", trust_remote_code=True)

# 2. Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# 3. Preprocessing function
def preprocess_function(examples):
    problems = examples["goal"]
    sol1s = examples["sol1"]
    sol2s = examples["sol2"]
    labels = examples["label"]

    inputs = []
    labels_list = []

    for i in range(len(problems)):
        problem = problems[i]
        sol1 = sol1s[i]
        sol2 = sol2s[i]
        label = labels[i]

        # Create two inputs per example
        inputs.extend([
            f"Problem: {problem} Solution 1: {sol1}",
            f"Problem: {problem} Solution 2: {sol2}"
        ])

        # Create labels for both solutions
        labels_list.extend([
            int(label == 0),  # 1 if sol1 is correct
            int(label == 1)   # 1 if sol2 is correct
        ])

    # Tokenize without returning tensors
    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels_list,
    }

# 4. Apply preprocessing to the dataset
tokenized_datasets = piqa_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['goal', 'sol1', 'sol2', 'label']  # Remove original columns
)

# 5. Example of accessing tokenized data
print(tokenized_datasets["train"][0])

# 6. Optional: Create a custom dataset class for DataLoader (if needed)

from torch.utils.data import Dataset

print(tokenized_datasets["train"])

# 7. Example of using DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=8, shuffle=False)

# Example to iterate through a dataloader.
example_batch = next(iter(train_loader))
print(example_batch)


tokenized_datasets = tokenized_datasets.with_format("torch")


# Hugging Face datasets need format conversion before DataLoader
# Replace:
# train_loader = DataLoader(tokenized_datasets["train"], ...)

# With:
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=8,
    collate_fn=data_collator,
    shuffle=True
)


from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments

from transformers import AutoModelForSequenceClassification

# Add classification head
model = GPT2ForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=2,
    pad_token_id=tokenizer.eos_token_id
)

# model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Convert to torch format
tokenized_datasets = tokenized_datasets.with_format("torch")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# Start training
trainer.train()
