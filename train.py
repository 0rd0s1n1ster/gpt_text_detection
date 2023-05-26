# %%
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
# %%
dataset = load_dataset("csv", data_files="df.csv")
ds = dataset['train'].remove_columns(["Unnamed: 0"])

# %%
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

ds = ds.map(preprocess_function, batched=True)
ds = ds.map(lambda x: {'label': [0 if i == 'orig' else 1 for i in x['label']]}, batched=True)
#%%
ds = ds.class_encode_column("label")
ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42, stratify_by_column="label")

# %%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# %%
import evaluate

accuracy = evaluate.load("accuracy")
# %%
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
# %%
id2label = {0: "orig", 1: "gpt"}
label2id = {"orig": 0, "gpt": 1}
# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)
# %%
training_args = TrainingArguments(
    output_dir="my_awesome_model_cleaner2",
    logging_dir='./logs_training2',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    # evaluation_strategy="epoch",
    eval_steps=20,
    # save_strategy="epoch",
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=20,
    load_best_model_at_end=True,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[TensorBoardCallback, EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=compute_metrics,
)
# %%
trainer.train()
trainer.save_model()
# %%
print(trainer.evaluate(ds['test']))
# %%
from transformers import pipeline
#%%
classifier = pipeline("sentiment-analysis", model="my_awesome_model_cleaner")
#classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
# %%
for id in range(10):
    text = ds['test'][id]['text']
    print(classifier(text),  ds['test'][id]['label'])
