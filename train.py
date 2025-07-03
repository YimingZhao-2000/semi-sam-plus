from config import Config
from data import get_dataloader
from model import get_student_model, get_teacher_model, update_ema
from utils import save_checkpoint, load_checkpoint, slice_volume, reassemble_mask, strong_aug, weak_aug, generate_pseudo_labels, variance_filter, export_mask, region_consistency_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =====================
# Training Phases
# =====================
def train_warmup(config, student, optimizer, train_loader, device):
    student.train()
    for epoch in range(config.warmup_epochs):
        for batch in train_loader:
            x_l, y_l = batch  # labeled data
            x_l, y_l = x_l.to(device), y_l.to(device)
            optimizer.zero_grad()
            # Only train decoder: freeze encoder
            for param in student.encoder.parameters():
                param.requires_grad = False
            output, loss = student(x_l, y_l)
            loss.backward()
            optimizer.step()
        print(f"[Warmup] Epoch {epoch+1}/{config.warmup_epochs} Loss: {loss.item():.4f}")

def train_semi(config, student, teacher, optimizer, train_loader, unlabeled_loader, device):
    epochs = config.semi_epochs
    student.train()
    teacher.eval()
    batches_per_epoch = min(len(train_loader), len(unlabeled_loader))
    max_iterations = epochs * batches_per_epoch
    # Incremental ramp-up/down on GPU
    lambda_c = torch.zeros(1, device=device)  # Consistency loss ramp-up
    lambda_s = torch.full((1,), 0.1, device=device)  # SAM loss ramp-down
    delta_c = torch.full((1,), 0.1 / max_iterations, device=device)
    delta_s = torch.full((1,), -0.1 / max_iterations, device=device)
    iteration = 0
    # Default mask prompt size for MedSAM2 pseudo label generation
    pseudo_label_size = getattr(config, 'pseudo_label_prompt_size', (32,32,32))
    for epoch in range(epochs):
        for (x_l, y_l), (x_u,) in zip(train_loader, unlabeled_loader):
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)
            optimizer.zero_grad()
            # --- Lsup: Supervised loss (Dice+CE) on labeled data ---
            out_l, loss_sup = student(x_l, y_l)
            # --- Lcon: Consistency loss (MSE) between student and teacher on unlabeled data ---
            with torch.no_grad():
                out_u_teacher, _ = teacher(x_u)
            out_u_student, _ = student(x_u)
            loss_con = F.mse_loss(out_u_student, out_u_teacher)
            # --- Lsam: Region consistency loss (pseudo-label region) ---
            with torch.no_grad():
                # Pass target_size to match MedSAM2 API and mask prompt structure
                pseudo_labels, mask_conf = generate_pseudo_labels(
                    teacher, x_u, out_u_student, t_strong=None, N=4, u_th=config.prompt_var_threshold, target_size=pseudo_label_size
                )
            loss_sam = region_consistency_loss(out_u_student, pseudo_labels, mask_conf)
            # --- Use ramp-up/down weights from GPU tensors ---
            loss = loss_sup + lambda_c.item() * loss_con + lambda_s.item() * loss_sam
            loss.backward()
            optimizer.step()
            # --- EMA update ---
            update_ema(student, teacher, config.ema_decay)
            # Increment ramp coefficients in-place on GPU
            lambda_c.add_(delta_c).clamp_(0, 0.1)
            lambda_s.add_(delta_s).clamp_(0, 0.1)
            iteration += 1
        print(f"[Semi] Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f} λc={lambda_c.item():.5f} λs={lambda_s.item():.5f}")

# =====================
# Main
# =====================
def main():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader(config, split='train')
    unlabeled_loader = get_dataloader(config, split='unlabeled')
    student = get_student_model(config)
    teacher = get_teacher_model(config, config.teacher_models[0])
    student.to(device)
    teacher.to(device)
    # Initialize teacher as a copy of student
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    optimizer = optim.Adam(student.parameters(), lr=config.lr_warmup)
    if config.phase == 'warmup':
        train_warmup(config, student, optimizer, train_loader, device)
    elif config.phase == 'semi':
        train_semi(config, student, teacher, optimizer, train_loader, unlabeled_loader, device)
    else:
        raise ValueError('Unknown training phase')
    save_checkpoint(student, optimizer, 0, config.checkpoint_dir + '/final.pth')

if __name__ == '__main__':
    main() 