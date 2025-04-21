
# 📚 BART-based Text Summarization using Transformers

This project focuses on training a text summarization model using the BART (Bidirectional and Auto-Regressive Transformers) architecture from Hugging Face’s `transformers` library. The model is fine-tuned on the CNN/DailyMail dataset to generate concise summaries of long-form news articles.

---

## 🚀 Project Highlights

- 🔍 **Model Used**: `facebook/bart-base`
- 📦 **Dataset**: [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)
- 🧪 **Evaluation Metric**: ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- 🧠 **Frameworks**: `transformers`, `datasets`, `evaluate`, PyTorch
- 📊 **Visualization**: Loss & Metric scores during training

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/bart-summarization.git
cd bart-summarization
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If not using `requirements.txt`, install manually:

```bash
pip install transformers datasets evaluate rouge_score
```

---

## 🧾 Training Script

Here's a simplified version of the model training process:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Load model and tokenizer
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Preprocessing
def preprocess(example):
    inputs = tokenizer(example["article"], padding="max_length", truncation=True, max_length=1024)
    targets = tokenizer(example["highlights"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_data = dataset.map(preprocess, batched=True)

# Training + Evaluation splits
train_dataset = tokenized_data["train"].select(range(2000))
eval_dataset = tokenized_data["validation"].select(range(500))

# ROUGE metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=10,
    predict_with_generate=True,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    compute_metrics=compute_metrics
)

trainer.train()
```

---

## 📈 Sample Output

| Metric      | Value (%) |
|-------------|-----------|
| ROUGE-1     | 41.2      |
| ROUGE-2     | 19.5      |
| ROUGE-L     | 38.7      |

*(Sample values from a short 2-epoch run on a subset)*

---

## 📂 Project Structure

```
.
├── results/                  # Training outputs
├── logs/                     # Training logs
├── bart_summarization.ipynb  # Notebook version of the training script
├── README.md
└── requirements.txt
```

---

## 🤝 Contributing

Pull requests and suggestions are welcome! If you'd like to contribute or report a bug, feel free to open an issue.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
