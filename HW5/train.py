from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils import compute_metrics
import nltk
nltk.download('punkt')

model_name = "t5-small"
max_input_length = 1024
max_target_length = 60
batch_size = 12
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
def preprocess_function(datas):
    for idx, article in enumerate(datas["body"]):
        if not article:
            datas["body"][idx] = ""
            datas["title"][idx] = ""
    inputs = ["summarize: " + doc for doc in datas["body"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(datas["title"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_dataset = load_dataset("json", data_files="./hw5_dataset/train.json",split = "train").train_test_split(test_size=0.05, shuffle=True)

args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-xsum",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=5*batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=raw_dataset['train'].map(preprocess_function, batched=True),
    eval_dataset=raw_dataset['test'].map(preprocess_function, batched=True),
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()