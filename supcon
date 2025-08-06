import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time
import random
import datetime
import os
import numpy as np
from skimage.feature import hog
import cv2
import math

TARGET_SIZE = (224, 224)
BATCH_SIZE = 50
FEATURE_DIM = 256
NUM_CLASSES = 2
LEARNING_RATE = 3e-5
EPOCHS = 50
EPSILON = 0.1
NUM_WORKERS = 4
LOG_INTERVAL = 10
TEMPERATURE = 0.07
ALPHA = 0.25
GAMMA = 2.0
DECAY_RATE = 0.3 
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

DATA_PATH = '/mnt/nas/public2/fengrongli/repos/deepfake_detection'
MODEL_PATH = '/mnt/nas/public2/fengrongli/repos/deepfake_detection/supcon/model_0713/new'

def create_run_dir(base_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(random.randint(1000, 9999))
    run_dir = f"run_{timestamp}_{random_id}"
    full_path = os.path.join(base_path, run_dir)
    os.makedirs(full_path, exist_ok=True)
    
    checkpoints_dir = os.path.join(full_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    tb_dir = os.path.join(full_path, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    
    return full_path, checkpoints_dir, tb_dir

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)

        # 构造相同类的mask: [B, B]
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)

        # 计算 cosine similarity 矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        # 数值稳定处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 去掉对角线（不和自己比）
        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # ----------- 👇 新增的关键处理逻辑 👇 ----------
        mask_sum = mask.sum(1)            # 每个样本有几个同类样本
        valid_mask = mask_sum > 1         # 只保留“有同类样本”的样本
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum - 1 + 1e-8)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss



class DeepFakeMultiBranchViT(nn.Module):
    def __init__(self, feature_dim=64, fusion='concat'):
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion = fusion

        # 初始化3个独立ViT主干（RGB, HOG, FFT）
        self.rgb_vit = self._init_vit()
        self.hog_vit = self._init_vit()
        self.fft_vit = self._init_vit()

        # 每个分支自己的 projection head（用于 SupCon）
        self.rgb_proj = self._projection_head()
        self.hog_proj = self._projection_head()
        self.fft_proj = self._projection_head()

        # 融合维度（3x768 if concat）
        fused_dim = 768 * 3 if fusion == 'concat' else 768
        self.classifier_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def _init_vit(self):
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity()
        return vit

    def _projection_head(self):
        return nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )

    def forward(self, rgb, hog, fft):
        # 分别提取特征
        feat_rgb = self.rgb_vit(rgb)     # (B, 768)
        feat_hog = self.hog_vit(hog)     # (B, 768)
        feat_fft = self.fft_vit(fft)     # (B, 768)

        # Projection head 用于 SupCon
        proj_rgb = self.rgb_proj(feat_rgb)  # (B, feature_dim)
        proj_hog = self.hog_proj(feat_hog)
        proj_fft = self.fft_proj(feat_fft)

        # 融合特征进行分类（Concat or Mean）
        if self.fusion == 'concat':
            fused = torch.cat([feat_rgb, feat_hog, feat_fft], dim=1)  # (B, 768*3)
        elif self.fusion == 'mean':
            fused = (feat_rgb + feat_hog + feat_fft) / 3
        else:
            raise ValueError("Unsupported fusion method")

        cls_out = self.classifier_head(fused).squeeze(1)  # (B,)

        return proj_rgb, proj_hog, proj_fft, cls_out


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float()  # ✅ 这行非常关键！

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class ExtraDataset(Dataset):
    def __init__(self, dataframe, image_size=(224, 224)):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size

        # 通用图像变换
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _compute_hog(self, image):
        # 转灰度并计算HOG
        img = np.array(image.convert("L")).astype(np.float32)

        # 计算 Sobel 梯度
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
        # 计算梯度幅值（magnitude）和方向（angle）
        mag, _ = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # 统一尺寸并构造 tensor
        mag = cv2.resize(mag, self.image_size)
        gx = cv2.resize(gx, self.image_size)
        gy = cv2.resize(gy, self.image_size)

        # 拼成 3 通道
        hog_tensor = np.stack([mag, gx, gy], axis=0)
        return torch.tensor(hog_tensor, dtype=torch.float32)

    def _compute_fft(self, image):
        image_np = np.array(image.convert("L"))  # 灰度
        f = np.fft.fft2(image_np)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
        magnitude = cv2.resize(magnitude, self.image_size)
        magnitude = np.stack([magnitude]*3, axis=0)  # 3通道
        return torch.tensor(magnitude, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        binary_label = int(row['label'])         # 0 or 1

        image = self._load_image(image_path)
        rgb = self.transform(image)
        hog = self._compute_hog(image)
        fft = self._compute_fft(image)

        return rgb, hog, fft, binary_label


class DeepFakeMultiModalDataset(Dataset):
    def __init__(self, dataframe, image_size=(224, 224)):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size

        # 通用图像变换
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _compute_hog(self, image):
        # 转灰度并计算HOG
        img = np.array(image.convert("L")).astype(np.float32)

        # 计算 Sobel 梯度
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
        # 计算梯度幅值（magnitude）和方向（angle）
        mag, _ = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # 统一尺寸并构造 tensor
        mag = cv2.resize(mag, self.image_size)
        gx = cv2.resize(gx, self.image_size)
        gy = cv2.resize(gy, self.image_size)

        # 拼成 3 通道
        hog_tensor = np.stack([mag, gx, gy], axis=0)
        return torch.tensor(hog_tensor, dtype=torch.float32)

    def _compute_fft(self, image):
        image_np = np.array(image.convert("L"))  # 灰度
        f = np.fft.fft2(image_np)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
        magnitude = cv2.resize(magnitude, self.image_size)
        magnitude = np.stack([magnitude]*3, axis=0)  # 3通道
        return torch.tensor(magnitude, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        binary_label = int(row['label'])         # 0 or 1
        multi_label = int(row['multi_label'])    # e.g., 0 ~ 5

        image = self._load_image(image_path)
        rgb = self.transform(image)
        hog = self._compute_hog(image)
        fft = self._compute_fft(image)

        return rgb, hog, fft, binary_label, multi_label


        
def get_lambda(current_step, total_steps):
    decay_steps = int(DECAY_RATE * total_steps)
    if current_step >= decay_steps:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * current_step / decay_steps))

def freeze_backbone(model):
    for param in model.rgb_vit.parameters():
        param.requires_grad = False
    for param in model.hog_vit.parameters():
        param.requires_grad = False
    for param in model.fft_vit.parameters():
        param.requires_grad = False
    for param in model.rgb_proj.parameters():
        param.requires_grad = False
    for param in model.hog_proj.parameters():
        param.requires_grad = False
    for param in model.fft_proj.parameters():
        param.requires_grad = False



def train_model():
    run_path, checkpoints_dir, tb_dir = create_run_dir(MODEL_PATH)
    print(f"Created run directory at: {run_path}")
    print(f"Checkpoints will be saved to: {checkpoints_dir}")
    print(f"TensorBoard logs will be saved to: {tb_dir}")

    # 初始化TensorBoard
    writer = SummaryWriter(tb_dir)

    # 加载数据
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'FF++_train_multilabel.csv'))
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'FF++_val_multilabel.csv'))

    train_set=DeepFakeMultiModalDataset(train_df)
    train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    val_set=DeepFakeMultiModalDataset(val_df)
    val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)

    model = DeepFakeMultiBranchViT(feature_dim=FEATURE_DIM,fusion='concat').to(DEVICE)
    supcon_loss = SupConLoss(temperature=TEMPERATURE)
    focal_loss = FocalLoss(alpha=ALPHA, gamma=GAMMA)
    

    best_accuracy = 0
    global_step = 0

    # 记录训练配置
    config_info = (f"Model Configuration:\n"
                  f"Batch Size: {BATCH_SIZE}\n"
                  f"Feature Dim: {FEATURE_DIM}\n"
                  f"Learning Rate: {LEARNING_RATE}\n"
                  f"Epsilon: {EPSILON}\n"
                  f"Device: {DEVICE}")
    
    writer.add_text('Configuration', config_info)

    for epoch in range(EPOCHS):
        epoch_dir = os.path.join(checkpoints_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        total_loss = 0
        sup_loss_rgb = 0
        sup_loss_hog = 0
        sup_loss_fft = 0
        class_loss = 0

        if epoch >= DECAY_RATE * EPOCHS:
            # 冻结特征提取部分（ViT 主干）
            # （可选）冻结 projection head（如果你不再用 supcon loss）
            freeze_backbone(model)
            
            optimizer = torch.optim.Adam(model.classifier_head.parameters(), lr=LEARNING_RATE)
            model.train()
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
            for batch_idx, (rgbs, hogs, ffts, labels, multi_labels) in enumerate(train_pbar):
                rgbs = rgbs.to(DEVICE, non_blocking=True)
                hogs = hogs.to(DEVICE, non_blocking=True)
                ffts = ffts.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                proj_rgb, proj_hog,proj_fft, cls_out= model(rgb=rgbs, hog=hogs, fft=ffts)
                loss = focal_loss(cls_out, labels)
                loss.backward()
                optimizer.step()
                class_loss += loss.item()

                writer.add_scalar('Step/Class_Loss', loss.item(), global_step)

                probs = torch.sigmoid(cls_out)
                pred_labels = (probs > 0.5).long()  # 0 或 1
                batch_correct = (pred_labels == labels.long()).sum().item()
                total = labels.size(0)
                batch_acc = batch_correct / total

                writer.add_scalar('Step/Train_Accuracy', batch_acc, global_step)
            
                train_pbar.set_postfix({
                    'cls_loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
            
                global_step += 1
            avg_class_loss = class_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Class Loss: {avg_class_loss:.4f}")
        
            # 记录每个epoch的平均损失
            writer.add_scalar('Epoch/Class_Loss', avg_class_loss, epoch)

            #验证
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            y_true = []
            y_scores = []

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
                for rgbs, hogs, ffts, labels, multi_labels in val_pbar:
                    rgbs = rgbs.to(DEVICE, non_blocking=True)
                    hogs = hogs.to(DEVICE, non_blocking=True)
                    ffts = ffts.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    multi_labels = multi_labels.to(DEVICE,non_blocking=True)

                    proj_rgb, proj_hog, proj_fft, cls_out= model(rgb=rgbs, hog=hogs, fft=ffts)

                    val_batch_loss = l*(supcon_loss(proj_rgb, multi_labels)+supcon_loss(proj_hog, multi_labels)+supcon_loss(proj_fft, multi_labels))+(1-l)* focal_loss(cls_out, labels)
                    val_loss += val_batch_loss.item()
  
                    probs = torch.sigmoid(cls_out)
                    pred_labels = (probs > 0.5).long()  # 0 或 1
                    batch_correct = (pred_labels == labels.long()).sum().item()
                    correct += batch_correct
                    total += labels.shape[0]

                    y_true.extend(labels.cpu().numpy())
                    y_scores.extend(probs.cpu().numpy())
                    current_acc = batch_correct / labels.shape[0]
                    val_pbar.set_postfix({
                        'loss': f'{val_batch_loss.item():.4f}',
                        'acc': f'{current_acc:.4f}'
                    })
            accuracy = correct / total
            avg_val_loss = val_loss / len(val_loader)
            # 计算 AUC
            try:
                auc_score = roc_auc_score(y_true, y_scores)
            except ValueError:
                auc_score = float('nan')  # 处理异常情况
        
            print(f"Validation - Accuracy: {accuracy:.4f}, Loss: {avg_val_loss:.4f}, AUC: {auc_score:.4f}")

            # 记录验证指标
            writer.add_scalar('Epoch/Val_Accuracy', accuracy, epoch)
            writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
            writer.add_scalar('Epoch/Val_AUC', auc_score, epoch)

        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            model.train()

            l = get_lambda(epoch,EPOCHS)
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
            for batch_idx, (rgbs, hogs, ffts, labels, multi_labels) in enumerate(train_pbar):
                rgbs = rgbs.to(DEVICE, non_blocking=True)
                hogs = hogs.to(DEVICE, non_blocking=True)
                ffts = ffts.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                multi_labels = multi_labels.to(DEVICE,non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                proj_rgb, proj_hog,proj_fft, cls_out= model(rgb=rgbs, hog=hogs, fft=ffts)

                # SupCon loss for each branch
                loss_supcon_rgb = supcon_loss(proj_rgb, multi_labels)
                loss_supcon_hog = supcon_loss(proj_hog, multi_labels)
                loss_supcon_fft = supcon_loss(proj_fft, multi_labels)

                # Focal BCE loss for final classifier
                loss_focal = focal_loss(cls_out, labels)

                loss = (l*(loss_supcon_rgb +
                       loss_supcon_hog +
                       loss_supcon_fft) +
                       (1 - l) * loss_focal)

                loss.backward()
                optimizer.step()

                sup_loss_rgb += loss_supcon_rgb.item()
                sup_loss_hog += loss_supcon_hog.item()
                sup_loss_fft += loss_supcon_fft.item()
                class_loss += loss_focal.item()
                total_loss += loss.item()

                writer.add_scalar('Step/supcon_Loss_rgb', loss_supcon_rgb.item(), global_step)
                writer.add_scalar('Step/supcon_Loss_hog', loss_supcon_hog.item(), global_step)
                writer.add_scalar('Step/supcon_Loss_fft', loss_supcon_fft.item(), global_step)
                writer.add_scalar('Step/Class_Loss', loss_focal.item(), global_step)
            
                probs = torch.sigmoid(cls_out)
                pred_labels = (probs > 0.5).long()  # 0 或 1
                batch_correct = (pred_labels == labels.long()).sum().item()
                total = labels.size(0)
                batch_acc = batch_correct / total

                writer.add_scalar('Step/Train_Accuracy', batch_acc, global_step)
            
                train_pbar.set_postfix({
                    'supcon_loss_rgb': f'{loss_supcon_rgb.item():.4f}',
                    'supcon_loss_hog': f'{loss_supcon_hog.item():.4f}',
                    'supcon_loss_fft': f'{loss_supcon_fft.item():.4f}',
                    'cls_loss': f'{loss_focal.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
            
                global_step += 1
            
            avg_loss = total_loss / len(train_loader)
            avg_supcon_loss_rgb = sup_loss_rgb / len(train_loader)
            avg_supcon_loss_hog = sup_loss_hog / len(train_loader)
            avg_supcon_loss_fft = sup_loss_fft / len(train_loader)
            avg_class_loss = class_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, SupCon Loss rgb: {avg_supcon_loss_rgb:.4f}, SupCon Loss hog: {avg_supcon_loss_hog:.4f}, SupCon Loss fft: {avg_supcon_loss_fft:.4f}, Class Loss: {avg_class_loss:.4f}")
        
            # 记录每个epoch的平均损失
            writer.add_scalar('Epoch/Total_Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/SupCon_Loss_RGB', avg_supcon_loss_rgb, epoch)
            writer.add_scalar('Epoch/SupCon_Loss_HOG', avg_supcon_loss_hog, epoch)
            writer.add_scalar('Epoch/SupCon_Loss_FFT', avg_supcon_loss_fft, epoch)
            writer.add_scalar('Epoch/Class_Loss', avg_class_loss, epoch)

            #验证
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            y_true = []
            y_scores = []

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
                for rgbs, hogs, ffts, labels, multi_labels in val_pbar:
                    rgbs = rgbs.to(DEVICE, non_blocking=True)
                    hogs = hogs.to(DEVICE, non_blocking=True)
                    ffts = ffts.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    multi_labels = multi_labels.to(DEVICE,non_blocking=True)

                    proj_rgb, proj_hog, proj_fft, cls_out= model(rgb=rgbs, hog=hogs, fft=ffts)

                    val_batch_loss = l*(supcon_loss(proj_rgb, multi_labels)+supcon_loss(proj_hog, multi_labels)+supcon_loss(proj_fft, multi_labels))+(1-l)* focal_loss(cls_out, labels)
                    val_loss += val_batch_loss.item()
  
                    probs = torch.sigmoid(cls_out)
                    pred_labels = (probs > 0.5).long()  # 0 或 1
                    batch_correct = (pred_labels == labels.long()).sum().item()
                    correct += batch_correct
                    total += labels.shape[0]

                    y_true.extend(labels.cpu().numpy())
                    y_scores.extend(probs.cpu().numpy())
                    current_acc = batch_correct / labels.shape[0]
                    val_pbar.set_postfix({
                        'loss': f'{val_batch_loss.item():.4f}',
                        'acc': f'{current_acc:.4f}'
                    })

            accuracy = correct / total
            avg_val_loss = val_loss / len(val_loader)
            # 计算 AUC
            try:
                auc_score = roc_auc_score(y_true, y_scores)
            except ValueError:
                auc_score = float('nan')  # 处理异常情况
        
            print(f"Validation - Accuracy: {accuracy:.4f}, Loss: {avg_val_loss:.4f}, AUC: {auc_score:.4f}")

            # 记录验证指标
            writer.add_scalar('Epoch/Val_Accuracy', accuracy, epoch)
            writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
            writer.add_scalar('Epoch/Val_AUC', auc_score, epoch)

        # 保存每个epoch的模型（仅保留最新2个）
        epoch_checkpoint_path = os.path.join(epoch_dir, f"checkpoint.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'auc': auc_score,
            'global_step': global_step
        }
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Epoch {epoch+1} model saved at {epoch_checkpoint_path}")

        # ✅ 管理 recent checkpoints：仅保留最近两个 epoch 的目录
        if 'recent_checkpoints' not in locals():
            recent_checkpoints = []
        recent_checkpoints.append(epoch_dir)

        if len(recent_checkpoints) > 2:
            old_ckpt_dir = recent_checkpoints.pop(0)
            try:
                import shutil
                shutil.rmtree(old_ckpt_dir)
                print(f"Deleted old checkpoint directory: {old_ckpt_dir}")
            except Exception as e:
                print(f"Warning: Failed to delete old checkpoint at {old_ckpt_dir}: {e}")

        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved at epoch {epoch+1} with accuracy {accuracy:.4f} and AUC {auc_score:.4f}")
            
            # 记录最佳模型信息
            writer.add_text('Best Model', f'Epoch: {epoch+1}, Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}', epoch)
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoints_dir, "final_model.pth")
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'auc': auc_score,
        'global_step': global_step
    }, final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    # 保存训练摘要到文本文件
    summary_path = os.path.join(run_path, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"===============\n")
        f.write(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Best Validation Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Final Validation Accuracy: {accuracy:.4f}\n")
        f.write(f"Final Validation AUC: {auc_score:.4f}\n")
        f.write(f"Model saved at: {checkpoints_dir}\n")
        f.write(f"TensorBoard logs: {tb_dir}\n")
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return best_accuracy, run_path

if __name__ == "__main__":
    # 设置PyTorch的内存分配器以提高性能
    torch.backends.cudnn.benchmark = True
    
    # 启用cudnn自动优化
    torch.backends.cudnn.enabled = True
    
    # 运行训练
    best_acc, run_dir = train_model()
    print(f"Training completed with best accuracy: {best_acc:.4f}")
    print(f"All outputs saved to: {run_dir}")




        












