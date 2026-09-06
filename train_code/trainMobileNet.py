import torch
import numpy as np
import evaluate
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

# 1. BƯỚC QUAN TRỌNG: Lọc sạch các mẫu có cropped_image bị None
print("Đang kiểm tra và lọc dữ liệu lỗi...")
clean_dataset = dataset["train"].filter(lambda example: example["cropped_image"] is not None)
print(f"Số lượng mẫu hợp lệ: {len(clean_dataset)} / {len(dataset['train'])}")

# 2. Khởi tạo Processor
model_id = "google/mobilenet_v2_1.0_224"
processor = AutoImageProcessor.from_pretrained(model_id)

# 3. Hàm transform có cơ chế dự phòng an toàn
def transform(batch):
    images = []
    for img in batch["cropped_image"]:
        if img is not None:
            images.append(img.convert("RGB"))
        else:
            # Ảnh đen dự phòng nếu lọt lưới
            images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = torch.tensor(batch["label"])
    return inputs

# 4. Chia tập train / val từ dataset đã được lọc sạch
split_ds = clean_dataset.train_test_split(test_size=0.2, seed=42)

train_data = split_ds["train"].with_transform(transform)
val_data = split_ds["test"].with_transform(transform)

# 5. Khởi tạo model và nhãn
id2label = {0: "live", 1: "spoof"}
label2id = {"live": 0, "spoof": 1}

model = AutoModelForImageClassification.from_pretrained(
    model_id,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 6. Hàm tính accuracy
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 7. TrainingArguments
training_args = TrainingArguments(
    output_dir="./antispoof_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to="none"
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
)

print("Bắt đầu huấn luyện...")
trainer.train()