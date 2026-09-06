"""
=============================================================================
Training Script: MiniFASNetV2 Anti-Spoofing
=============================================================================
Huấn luyện mô hình MiniFASNetV2 phân loại khuôn mặt Real / Fake (Spoof).

Kiến trúc model KHỚP CHÍNH XÁC với code inference trong:
  src/anti_spoof/minifasnet.py

Label mapping (khớp với inference):
  - Class 0 = FAKE / SPOOF (ảnh in, màn hình, video phát lại)
  - Class 1 = REAL / LIVE  (khuôn mặt thật)

Dataset hỗ trợ:
  - CelebA-Spoof (cấu trúc: .../live/xxx.png và .../spoof/xxx.png)
  - Bất kỳ dataset nào có cấu trúc thư mục live/ vs spoof/

Chạy trên:
  - Kaggle Notebook (GPU P100/T4)
  - Google Colab (GPU T4/A100)
  - Local (CPU hoặc CUDA GPU)

Tính năng:
  - Mixed Precision (AMP) cho tốc độ training nhanh hơn trên GPU
  - Label Smoothing giúp generalize tốt hơn
  - Auto class weights cho imbalanced dataset
  - Resume training từ checkpoint
  - Visualization: loss curves, accuracy, confusion matrix, val batch samples
  - Export model tương thích 100% với inference code
=============================================================================
"""

import os
import glob
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend cho server/notebook
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

# Bỏ qua lỗi ảnh bị cắt cụt / thiếu byte
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==============================================================
# 1. KIẾN TRÚC MINIFASNETV2 (KHỚP VỚI INFERENCE)
# ==============================================================
# ⚠️ CÁC CLASS DƯỚI ĐÂY PHẢI GIỐNG HỆT file:
#    src/anti_spoof/minifasnet.py
# Nếu thay đổi inference, phải cập nhật lại đây!
# ==============================================================

class Conv_block(nn.Module):
    """Standard Convolution + BatchNorm + PReLU block with grouping support"""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class Linear_block(nn.Module):
    """Convolution + BatchNorm (no activation)"""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class Depth_Wise(nn.Module):
    """DepthWise Separable Convolution block"""
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.residual = residual
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x):
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            return short_cut + x
        return x


class Residual(nn.Module):
    """Residual block containing multiple Conv_blocks"""
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        modules = [
            Conv_block(c, c, kernel=kernel, stride=stride, padding=padding, groups=groups)
            for _ in range(num_block)
        ]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MiniFASNetV2(nn.Module):
    """
    MiniFASNetV2 (Silent-Face-Anti-Spoofing lightweight backbone)
    Classes: 0 = FAKE / SPOOF, 1 = REAL
    """
    def __init__(self, embedding_size=128, num_classes=2, img_channel=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.conv2 = Conv_block(32, 64, kernel=3, stride=1, padding=1, groups=32)
        self.res1 = Residual(64, 1, groups=64)
        self.conv3 = Conv_block(64, 128, kernel=3, stride=2, padding=1, groups=64)
        self.res2 = Residual(128, 2, groups=128)
        self.conv4 = Conv_block(128, 128, kernel=3, stride=2, padding=1, groups=128)
        self.res3 = Residual(128, 2, groups=128)
        self.conv5 = Conv_block(128, 256, kernel=3, stride=2, padding=1, groups=128)
        self.res4 = Residual(256, 1, groups=256)
        self.fc = nn.Linear(256, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        x = self.conv5(x)
        x = self.res4(x)
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        feat = self.fc(x)
        return self.classifier(feat)


# ==============================================================
# 2. DATASET LOADER
# ==============================================================
def is_live_image(img_path):
    """
    Xác định ảnh thuộc class Live hay Spoof dựa trên path.
    Xử lý cả Windows (\\) và Linux (/) path separator.
    """
    normalized = img_path.replace('\\', '/')
    if '/live/' in normalized or '/Live/' in normalized or '/real/' in normalized or '/Real/' in normalized:
        return True
    return False


class AntiSpoofDataset(Dataset):
    """
    Dataset loader cho CelebA-Spoof hoặc dataset tương tự.
    Label mapping KHỚP với inference:
      - 0 = FAKE / SPOOF
      - 1 = REAL / LIVE
    """
    def __init__(self, image_paths, transform=None, use_face_crop=True):
        self.image_paths = image_paths
        self.transform = transform
        self.use_face_crop = use_face_crop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Label: 1=REAL, 0=FAKE (KHỚP VỚI INFERENCE)
        label = 1 if is_live_image(img_path) else 0

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            new_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(new_idx)

        if self.use_face_crop:
            img = self._crop_face(img, img_path)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def _crop_face(self, img, img_path, scale=2.7):
        """
        Cắt khuôn mặt theo Bounding Box với scale mở rộng (mặc định 2.7x)
        nhằm lấy thêm ngữ cảnh cổ, vai và viền thiết bị/màn hình.
        """
        bb_path = img_path.replace('.png', '_BB.txt').replace('.jpg', '_BB.txt')
        if not os.path.exists(bb_path):
            return img

        try:
            w_img, h_img = img.size
            with open(bb_path, 'r') as f:
                coords = [int(x) for x in f.readline().split()[:4]]
                x, y, w, h = coords
                if w <= 0 or h <= 0:
                    return img

                # Giới hạn scale tối đa trong khung ảnh
                s = min((h_img - 1) / float(h), min((w_img - 1) / float(w), float(scale)))
                w_new, h_new = w * s, h * s
                cx, cy = x + w / 2.0, y + h / 2.0

                left_top_x = cx - w_new / 2.0
                left_top_y = cy - h_new / 2.0
                right_bottom_x = cx + w_new / 2.0
                right_bottom_y = cy + h_new / 2.0

                # Trượt cửa sổ crop nếu chạm viền ảnh
                if left_top_x < 0:
                    right_bottom_x -= left_top_x
                    left_top_x = 0
                if left_top_y < 0:
                    right_bottom_y -= left_top_y
                    left_top_y = 0
                if right_bottom_x > w_img - 1:
                    left_top_x -= (right_bottom_x - w_img + 1)
                    right_bottom_x = w_img - 1
                if right_bottom_y > h_img - 1:
                    left_top_y -= (right_bottom_y - h_img + 1)
                    right_bottom_y = h_img - 1

                x1 = max(0, int(round(left_top_x)))
                y1 = max(0, int(round(left_top_y)))
                x2 = min(w_img, int(round(right_bottom_x)))
                y2 = min(h_img, int(round(right_bottom_y)))

                if x2 > x1 and y2 > y1:
                    img = img.crop((x1, y1, x2, y2))
        except Exception:
            pass

        return img


# ==============================================================
# 3. VISUALIZATION HELPERS
# ==============================================================
def plot_val_batch(images, labels, preds, epoch, save_dir):
    num_samples = min(16, len(images))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    class_names = ["FAKE", "REAL"]

    for i in range(16):
        ax = axes[i]
        if i < num_samples:
            img = images[i].cpu().numpy().transpose((1, 2, 0))
            img = np.clip(std * img + mean, 0, 1)
            lbl, prd = labels[i].item(), preds[i].item()
            ax.imshow(img)
            color = "green" if lbl == prd else "red"
            ax.set_title(
                f"True: {class_names[lbl]}\nPred: {class_names[prd]}",
                color=color, fontsize=9, fontweight='bold'
            )
        ax.axis('off')

    plt.suptitle(f"Validation Batch - Epoch {epoch + 1}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"val_batch_epoch_{epoch + 1}.png"), dpi=150)
    plt.close()


def plot_training_metrics(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
    axes[0].set_title('Loss Curve', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Train Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'r-o', markersize=4, label='Val Accuracy')
    axes[1].set_title('Accuracy Curve (%)', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "results.png"), dpi=150)
    plt.close()


def plot_confusion_matrix_chart(all_labels, all_preds, save_dir):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['FAKE (0)', 'REAL (1)'],
        yticklabels=['FAKE (0)', 'REAL (1)']
    )
    plt.title('Validation Confusion Matrix', fontweight='bold', fontsize=13)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


# ==============================================================
# 4. TRAINING LOOP
# ==============================================================
def train_model():
    # ========================
    # CONFIGURATION
    # ========================
    EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 7
    LABEL_SMOOTHING = 0.05
    INPUT_SIZE = (80, 80)   # Khop voi inference preprocess_crop()
    TEST_SIZE = 0.2
    USE_AMP = True
    RESUME = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        USE_AMP = False
    print(f"\n{'='*70}")
    print(f"  MINIFASNETV2 TRAINING - Anti-Spoofing")
    print(f"{'='*70}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch Size   : {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Input Size   : {INPUT_SIZE}")
    print(f"  AMP          : {USE_AMP}")
    print(f"  Label Smooth : {LABEL_SMOOTHING}")
    print(f"{'='*70}\n")

    # ========================
    # OUTPUT DIRECTORIES
    # ========================
    if os.path.exists("/kaggle"):
        save_dir = "/kaggle/working/runs/train"
    elif os.path.exists("/content"):
        save_dir = "/content/runs/train"
    else:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "train")
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "best_minifasnetv2.pth")
    last_checkpoint_path = os.path.join(save_dir, "last_checkpoint.pth")
    export_model_path = os.path.join(save_dir, "Anti_Spoof_minifasnetv2.pth")

    print(f"[INFO] Save directory: {save_dir}")

    # ========================
    # SCAN DATASET
    # ========================
    if os.path.exists("/kaggle/input"):
        input_root = "/kaggle/input"
    elif os.path.exists("/content/dataset"):
        input_root = "/content/dataset"
    else:
        input_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    print(f"[INFO] Scanning images from: {input_root}")
    all_images = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        all_images.extend(glob.glob(os.path.join(input_root, "**", ext), recursive=True))

    if not all_images:
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            all_images.extend(glob.glob(os.path.join("/content", "**", ext), recursive=True))

    print(f"[INFO] Total images found: {len(all_images)}")
    if len(all_images) == 0:
        print("[ERROR] Khong tim thay anh nao! Kiem tra lai duong dan dataset.")
        return

    num_live = sum(1 for p in all_images if is_live_image(p))
    num_spoof = len(all_images) - num_live
    print(f"[INFO] Distribution: REAL/Live = {num_live} | FAKE/Spoof = {num_spoof}")

    if num_live == 0 or num_spoof == 0:
        print("[WARNING] Dataset chi co 1 class! Kiem tra lai cau truc thu muc (can co /live/ va /spoof/).")

    train_paths, val_paths = train_test_split(
        all_images, test_size=TEST_SIZE, random_state=42, shuffle=True
    )
    print(f"[INFO] Train: {len(train_paths)} | Val: {len(val_paths)}")

    # ========================
    # CLASS WEIGHTS (auto)
    # ========================
    num_live_train = sum(1 for p in train_paths if is_live_image(p))
    num_spoof_train = len(train_paths) - num_live_train
    w_fake = 1.0 / max(num_spoof_train, 1)
    w_real = 1.0 / max(num_live_train, 1)
    weights = torch.tensor([w_fake, w_real], dtype=torch.float).to(device)
    weights = weights / weights.sum()
    print(f"[INFO] Class weights: FAKE={weights[0]:.4f}, REAL={weights[1]:.4f}")

    # ========================
    # DATA AUGMENTATION
    # ========================
    train_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ========================
    # DATA LOADERS
    # ========================
    num_workers = 4 if os.path.exists("/kaggle") else 2
    train_loader = DataLoader(
        AntiSpoofDataset(train_paths, train_transform),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        AntiSpoofDataset(val_paths, val_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # ========================
    # MODEL, LOSS, OPTIMIZER
    # ========================
    model = MiniFASNetV2(embedding_size=128, num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)

    # ========================
    # RESUME FROM CHECKPOINT
    # ========================
    start_epoch = 0
    best_acc = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    if RESUME and os.path.exists(last_checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {last_checkpoint_path}")
        ckpt = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        history = ckpt.get('history', history)
        no_improve = ckpt.get('no_improve', 0)
        print(f"[INFO] Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")
    else:
        print("[INFO] Training from scratch (new model)")

    # ========================
    # TRAINING LOOP
    # ========================
    print(f"\n{'='*70}")
    print(f"  BAT DAU HUAN LUYEN TU EPOCH {start_epoch + 1}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                running_acc = train_correct / total_train * 100
                print(f"  Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {running_acc:.1f}%")

        scheduler.step()
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (train_correct / total_train) * 100

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        all_val_labels = []
        all_val_preds = []
        sample_batch = None

        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(enabled=USE_AMP):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

                if i == 0:
                    sample_batch = (images, labels, preds)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (val_correct / total_val) * 100

        # --- LOG ---
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        lr_now = optimizer.param_groups[0]['lr']
        time_now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{time_now}] Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"| Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% "
              f"| LR: {lr_now:.6f}")

        # --- SAVE CHECKPOINT ---
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'no_improve': no_improve,
            'history': history,
        }
        torch.save(checkpoint_data, last_checkpoint_path)

        # --- VISUALIZATION ---
        if sample_batch:
            plot_val_batch(sample_batch[0], sample_batch[1], sample_batch[2], epoch, save_dir)
        plot_training_metrics(history, save_dir)

        # --- BEST MODEL ---
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            no_improve = 0

            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), export_model_path)

            plot_confusion_matrix_chart(all_val_labels, all_val_preds, save_dir)

            report = classification_report(
                all_val_labels, all_val_preds,
                target_names=['FAKE (0)', 'REAL (1)'],
                digits=4
            )
            with open(os.path.join(save_dir, "classification_report.txt"), 'w') as f:
                f.write(f"Epoch {epoch+1} - Best Val Acc: {best_acc:.2f}%\n\n")
                f.write(report)

            print(f"  >>> [BEST] Da luu Best Model ({best_acc:.2f}%)")
            print(f"      -> {best_model_path}")
            print(f"      -> {export_model_path} (inference-compatible)")
        else:
            no_improve += 1
            print(f"  (No improve: {no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print(f"\n[EARLY STOPPING] Dung som tai epoch {epoch+1} "
                      f"sau {PATIENCE} epochs khong cai thien.")
                break

    # ========================
    # SUMMARY
    # ========================
    print(f"\n{'='*70}")
    print(f"  HUAN LUYEN HOAN TAT!")
    print(f"{'='*70}")
    print(f"  Best Val Accuracy : {best_acc:.2f}%")
    print(f"  Total Epochs      : {epoch + 1}")
    print(f"  Save directory    : {save_dir}")
    print(f"  Best model        : {best_model_path}")
    print(f"  Export model      : {export_model_path}")
    print(f"")
    print(f"  De su dung model trong inference:")
    print(f"    1. Copy file '{os.path.basename(export_model_path)}' vao thu muc models/")
    print(f"    2. Doi ten thanh 'Anti_Spoof_minifasnetv2_(X).pth'")
    print(f"    3. Code inference se tu dong tim va load model moi nhat.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    train_model()
