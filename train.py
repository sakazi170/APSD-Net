
import os
import tqdm
import argparse
import setproctitle
import time
import csv
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.utils import one_hot
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
from utils.train_data_loader import SubjectReader
from utils.iterator import set_random_seed, CosineAnnealingWithWarmUp
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from Networks import (unet, BTS_Baseline_VIT)
from Networks_bts import BTS_t2
from Networks_bts import BTS_flair
from Networks_bts import BTS_t1
from Networks_bts import BTS_t1ce
from Networks_freq import BTS_FLGFA, BTS_FLGFA1, BTS_FLGFA2, BTS_FLGFA3
from Networks_cmbf import BTS_CMBF, BTS_CMBF1, BTS_CMBF2, BTS_CMBF3
from Networks_base import BTS_Baseline
from demo import BTS_new

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline1', help='network for training')
    parser.add_argument('--dataset', default="brats2020", help='dataset name')
    parser.add_argument('--interval', type=int, default=1, help='interval for validation')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs')
    parser.add_argument('--gpu', type=str, default="3", help='GPU device id')
    parser.add_argument('--ncpu', type=int, default=4, help='number of workers')
    parser.add_argument('--bsize', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='maximum epochs for training')

    parser.add_argument('--trainset', action='store_true', help='use full training set')
    parser.add_argument('--mixed', action="store_true", help='whether to use mixed precision training')
    parser.add_argument('--benchmark', action="store_true", default=False, help='Enable cudnn benchmark for faster training')
    parser.add_argument('--verbose', action='store_true', help='show progress bar')

    parser.add_argument('--cp', type=str, help='model checkpoint for continuation')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type')
    parser.add_argument('-cfg', type=str, default='adam.cfg', help='config file')

    parser.add_argument('--loss', type=str, default='both', choices=['dice', 'both'], help='Loss function type')
    parser.add_argument('--dice', type=float, default=1.0, help='Dice loss weight')
    parser.add_argument('--ce', type=float, default=1.0, help='Cross-entropy loss weight')

    return parser.parse_args()

def get_loss_function(args):
    # Option 1 — Pure Dice Loss
    if args.loss == "dice":
        print("Using: Dice Loss Only")
        return DiceLoss(to_onehot_y=False, softmax=True, squared_pred=True)
    # Option 2 — Dice + CE (custom weights)
    if args.loss == "both":
        print(f"Using: Dice + CE (Weighted)  Dice={args.dice}  CE={args.ce}")
        return DiceCELoss(to_onehot_y=False, softmax=True, squared_pred=True, lambda_dice=args.dice, lambda_ce=args.ce)

    raise ValueError("Invalid loss type!")


def main():
    start_time = time.time()
    args = parse_args()
    torch.backends.cudnn.benchmark = args.benchmark
    model_dict = {
        'unet': unet,
        'BTS_t2': BTS_t2,
        'BTS_FLGFA': BTS_FLGFA,
        'BTS_FLGFA1': BTS_FLGFA1,
        'BTS_FLGFA2': BTS_FLGFA2,
        'BTS_FLGFA3': BTS_FLGFA3,
        'BTS_CMBF': BTS_CMBF,
        'BTS_CMBF1': BTS_CMBF1,
        'BTS_CMBF2': BTS_CMBF2,
        'BTS_CMBF3': BTS_CMBF3,
        'BTS_Baseline': BTS_Baseline,
        'BTS_Baseline_VIT': BTS_Baseline_VIT,
        'BTS_new': BTS_new,
        'BTS_flair': BTS_flair,
        'BTS_t1': BTS_t1,
        'BTS_t1ce': BTS_t1ce,
    }
    assert args.model in model_dict.keys(), 'Model name is wrong!'

    # Extract arguments
    model_name = args.model
    dataset_name = args.dataset
    is_mixed = args.mixed
    benchmark = args.benchmark
    num_gpu = args.ngpu
    gpu_id = args.gpu
    num_workers = args.ncpu
    use_trainset = args.trainset
    verbose = args.verbose

    if args.dataset == 'brats2019':
        data_root = "/data/qazisami/dataset/BraTS2019/train"
    elif dataset_name == 'brats2020':
        data_root = "/data/qazisami/dataset/BraTS2020/training80"
    elif dataset_name == 'brats2021':
        data_root = "/data/qazisami/dataset/BraTS2023-GLI/BraTS2023-GLI-TrainingData"
    elif dataset_name == 'brats2023':
        data_root = "/data/qazisami/dataset/BraTS2023-MEN/BraTS_MEN_Train/train"
    else:
        raise Exception('[error]please provide correct dataset name!')


    if args.loss == "dice":
        save_dir = f'./checkpoints/{dataset_name}/{model_name}_{args.loss}'
        os.makedirs(save_dir, exist_ok=True)
        save_dir_logs = f'./logs/{dataset_name}/{model_name}_{args.loss}'
        os.makedirs(save_dir_logs, exist_ok=True)
    elif args.loss == "both":
        save_dir = f'./checkpoints/{dataset_name}/{model_name}_{args.loss}_{args.dice}_{args.ce}'
        os.makedirs(save_dir, exist_ok=True)
        save_dir_logs = f'./logs/{dataset_name}/{model_name}_{args.loss}_{args.dice}_{args.ce}'
        os.makedirs(save_dir_logs, exist_ok=True)

    # Training parameters
    seed = 42
    batch_size = args.bsize
    num_epochs = args.epochs
    lr = 0.001
    decay = 0.00001
    warm_up_epochs = 5
    max_lr_epochs = 50
    patch_size = 128

    print('-' * 100)
    print(f'{dataset_name} Challenge Training!')
    print(f'Model: {model_name}')
    print(f'Mixed Precision - {is_mixed};  \nCUDNN Benchmark - {benchmark};  \nGPU ID - {gpu_id};  \nBatch Size - {batch_size};  \nWorkers - {num_workers}')

    # Set random seed
    set_random_seed(seed=seed, benchmark=benchmark)

    # Initialize SubjectReader with default paths
    subject_reader = SubjectReader(train_dir=data_root, train_dataset=dataset_name, training_size=patch_size)


    if use_trainset:
        print(f"Using full training set from {data_root}")
        trainset = subject_reader.get_trainset()

    # Create training loader
    train_loader = DataLoader(trainset, batch_size=batch_size * num_gpu, shuffle=True,
                              num_workers=num_workers, multiprocessing_context='spawn')

    # Setup GPU
    gpu_ids = [int(id) for id in args.gpu.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    if args.model == 'unet':
        model = model_dict[args.model](patch_size, patch_size, patch_size, in_channels=4, num_classes=4).to(device)
    else:
        model = model_dict[args.model](patch_size, patch_size, patch_size, in_channels=1, num_classes=4).to(device)

    # Setup loss, optimizer and scheduler
    criterion = get_loss_function(args)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=decay)
    scheduler = CosineAnnealingWithWarmUp(optimizer,
                                          cycle_steps=num_epochs * len(train_loader),
                                          max_lr_steps=max_lr_epochs * len(train_loader),
                                          max_lr=lr,
                                          min_lr=lr / 1000,
                                          warmup_steps=warm_up_epochs * len(train_loader))

    # Load checkpoint if provided
    start_epoch = 0
    if args.cp:
        checkpoint = torch.load(args.cp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    # Multi-GPU setup if needed
    if num_gpu > 1:
        print(f'Multi GPU not supported.')
        sys.exit()

    scaler = torch.amp.GradScaler('cuda')

#======================== Training Loop ==================================================
    best_loss = 2.0


    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(save_dir_logs, f"{model_name}_{args.loss}_{date_str}.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss"])

    previous_best_model = None  # To track the previous best model filename
    total_steps = len(train_loader)
    steps_per_print = total_steps // 2  # This will give you 2 prints per epoch
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('-' * 100)

    # Training loop
    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_start_time = time.time()
        setproctitle.setproctitle('{}: {}/{}'.format(model_name, epoch, num_epochs))
        print(f"\nEpoch {epoch}/{num_epochs}")
        model.train()
        epoch_loss = 0
        loader = tqdm.tqdm(train_loader) if verbose else train_loader

        # Training phase
        for step, batch_data in enumerate(loader):
            inputs_t1 = batch_data['t1'].to(device)
            inputs_t1ce = batch_data['t1ce'].to(device)
            inputs_t2 = batch_data['t2'].to(device)
            inputs_flair = batch_data['flair'].to(device)
            targets = one_hot(batch_data['label'].to(device), num_classes=4)

            optimizer.zero_grad()

            if is_mixed:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs_t1, inputs_t1ce, inputs_t2, inputs_flair)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # ← ADD
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← ADD
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs_t1, inputs_t1ce, inputs_t2, inputs_flair)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← ADD
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            step_loss = loss.item()  # Current step loss

            if verbose:
                loader.set_postfix_str(f'lr:{optimizer.param_groups[0]["lr"]:.8f} - loss:{step_loss:.6f}')
            elif (step + 1) % steps_per_print == 0:
                print(f'Step {step + 1}/{total_steps} - lr:{optimizer.param_groups[0]["lr"]:.8f} - loss:{step_loss:.4f}')

        epoch_end_time = time.time()
        epoch_duration = (epoch_end_time - epoch_start_time) / 60
        final_avg_loss = epoch_loss / total_steps  # Calculate average loss at epoch end
        print(f"Epoch Average Loss: {final_avg_loss:.4f} Time: {epoch_duration:.2f} minutes")

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{final_avg_loss:.5f}"])

        if final_avg_loss < best_loss:
            # Store old model name before updating best_loss
            old_model_name = previous_best_model
            best_loss = final_avg_loss
            best_checkpoint = {
                "net": model.module.state_dict() if num_gpu > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch,
                "loss": best_loss
            }
            # Create new model name
            best_model_name = f'{model_name}_best_model_{epoch}_{best_loss:.4f}.pkl'
            # Save new model first before deleting old one (safer)
            try:
                torch.save(best_checkpoint, os.path.join(save_dir, best_model_name))
                # Only delete old model after successful save
                if old_model_name and os.path.exists(os.path.join(save_dir, old_model_name)):
                    try:
                        os.remove(os.path.join(save_dir, old_model_name))
                    except OSError as e:
                        print(f"Warning: Could not delete previous model: {e}")
                # Update tracking variable after successful operations
                previous_best_model = best_model_name
                print(f"Best model saved at epoch {epoch}! loss: {best_loss:.4f}")
            except Exception as e:
                print(f"Error saving new best model: {e}")

        # Regular checkpoint saving (every 10 epochs)
        if epoch % 50 == 0:
            checkpoint = {
                "net": model.module.state_dict() if num_gpu > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch,
                "loss": final_avg_loss
            }
            checkpoint_name = f'{model_name}_{epoch}_{final_avg_loss:.4f}.pkl'
            torch.save(checkpoint, os.path.join(save_dir, checkpoint_name))
            print(f"Saved checkpoint at epoch {epoch}")

        # Final total time after training completes
    total_time = (time.time() - start_time) / 60
    print(f"Training completed. Total time: {total_time:.2f} minutes")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()