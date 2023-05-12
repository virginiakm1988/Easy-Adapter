from datasets import load_dataset
from adapter_bert import adapted_bert_output
from utils import mark_only_adapter_as_trainable, TrainingArgumentsWithMPSSupport
from transformers import AutoTokenizer,TrainingArguments, Trainer, AutoModelForSequenceClassification


raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

model_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model_bert.config.adapter = "houlsby"
original_state_dict = model_bert.state_dict()
for idx, layer in enumerate(model_bert.bert.encoder.layer):
  model_bert.bert.encoder.layer[idx].output = adapted_bert_output(model_bert.bert.encoder.layer[idx].output, model_bert.config)
#freeze parameters
model_bert.load_state_dict(original_state_dict,strict = False)
mark_only_adapter_as_trainable(model_bert)

training_args = TrainingArgumentsWithMPSSupport("test_trainer")

trainer = Trainer(
    model=model_bert, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
)

trainer.train()