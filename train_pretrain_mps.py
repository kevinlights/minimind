import os
import time
import math
import argparse
import warnings
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

from model.model import MiniMindLM  # 假设这是你的自定义模型
from model.LMConfig import LMConfig  # 假设这是模型配置
from model.dataset import PretrainDataset  # 假设这是数据集类

warnings.filterwarnings('ignore')

# 日志函数（简化版）
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print(f"当前 GPU 内存占用: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
    print(f"驱动程序内存占用: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")

# 学习率调度（余弦退火）
def get_lr(current_step, total_steps, base_lr):
    return base_lr / 10 + 0.5 * base_lr * (1 + math.cos(math.pi * current_step / total_steps))

# 训练单个epoch
def train_epoch(model, loader, optimizer, epoch, args):
    model.train()
    loss_func = nn.CrossEntropyLoss(reduction='none')
    total_loss = 0
    start_time = time.time()
    
    for step, (X, Y, mask) in enumerate(loader):
        # 数据移至设备
        X, Y, mask = X.to(args.device), Y.to(args.device), mask.to(args.device)
        
        # 动态学习率
        current_step = epoch * len(loader) + step
        lr = get_lr(current_step, args.epochs * len(loader), args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 前向计算（无autocast）
        outputs = model(X)
        loss = loss_func(outputs.logits.view(-1, outputs.logits.size(-1)), Y.view(-1))
        loss = (loss * mask.view(-1)).sum() / mask.sum()
        
        # 反向传播（无GradScaler）
        loss.backward()
        
        # 梯度累积
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        # 日志记录
        if step % args.log_interval == 0:
            avg_time = (time.time() - start_time) / (step + 1)
            remaining = (len(loader) - step) * avg_time / 60
            log(f"Epoch {epoch+1}/{args.epochs} | Step {step}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                f"ETA: {remaining:.1f}min")

    return total_loss / len(loader)

# 初始化模型
def init_model(config, device):
    model = MiniMindLM(config).to(device)
    log(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model

if __name__ == "__main__":
    # 参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./dataset/pretrain_hq_mini.jsonl")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)  # M4建议较小batch
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="./out")
    args = parser.parse_args()

    # 设备设置（自动检测MPS）
    args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"使用设备: {args.device.upper()}")

    # 初始化模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/minimind_tokenizer/")
    config = LMConfig(
        dim=512,
        n_layers=8,
        max_seq_len=512
    )
    model = init_model(config, args.device)

    # 数据集和数据加载器（优化CPU使用）
    train_data = PretrainDataset(args.data, tokenizer, max_length=config.max_seq_len)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # Mac建议2-4个workers
        pin_memory=False,  # MPS不需要pin_memory
        persistent_workers=True  # 减少重复初始化
    )

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 训练循环
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, args)
        log(f"Epoch {epoch+1} 平均损失: {train_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 1 == 0:  # 每个epoch保存一次
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.save_dir}/epoch_{epoch+1}.pt")
            log(f"检查点已保存至 {args.save_dir}/epoch_{epoch+1}.pt")

    log("训练完成！")