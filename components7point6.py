from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from datasets import load_dataset, DatasetDict
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
context_length = 128
## load and preprocess dataset
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
raw_datasets = DatasetDict(
    {
        "train": ds_train.shuffle().select(range(5000)), # originally .select(range(50000))
        "valid": ds_valid.shuffle().select(range(1000))  # originally .select(range(500))
    }
)
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}
tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
## instantiate model
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
model = GPT2LMHeadModel(config)
