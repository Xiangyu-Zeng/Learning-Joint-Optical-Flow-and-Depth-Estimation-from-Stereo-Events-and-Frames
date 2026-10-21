import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import logging
import time
import h5py
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def ensure_init_files():
    init_paths = [
        os.path.join(PROJECT_ROOT, 'datasets', '__init__.py'),
        os.path.join(PROJECT_ROOT, 'datasets', 'MVSEC', '__init__.py')
    ]
    for path in init_paths:
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write('# Package marker\n')

ensure_init_files()

from datasets.MVSEC.MVSECProvider import MVSECProvider
from EventDepthFlowNet import EventDepthFlowNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MVSECWithDepthDatasetRGB(Dataset):
    def __init__(self, hdf5_data_path, hdf5_gt_path, sequence='indoor_flying1', dt=1, bins=15, crop_size=(240,320)):
        self.h, self.w = crop_size
        self.sequence = sequence
        data_root = os.path.dirname(os.path.dirname(hdf5_data_path))
        config = {
            'name': 'mvsec',
            'task': 'flow',
            'SequenceClass': 'MVSECSequence_flow_with_image',
            'SequenceNames': [sequence],
            'root_path': data_root,
            'split': 'test',
            'location': 'left',
            'dt': dt,
            'voxel_bins': bins,
            'crop_size': [self.h, self.w],
            'augment': False
        }
        provider = MVSECProvider(config)
        self.mvsec_sequences, _ = provider.get_datasets(F_list=True)
        self.flow_dataset = provider.get_datasets()

        self.start_no = self.mvsec_sequences[0].start_no
        self.length = len(self.flow_dataset)

        self.gt_file = h5py.File(hdf5_gt_path, 'r')
        self.depth = self.gt_file['davis/left/depth_image_raw'][:]
        total_depth = len(self.depth)
        available_depth = total_depth - self.start_no

        if available_depth > 0 and self.start_no < total_depth:
            self.length = min(len(self.flow_dataset), available_depth)
            self.use_start_no = True
        else:
            self.length = min(len(self.flow_dataset), total_depth)
            self.start_no = 0
            self.use_start_no = False

        logging.info(f'Loaded {self.length} samples with RGB and depth for sequence {sequence}')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        flow_sample = self.flow_dataset[idx]
        left_voxel = flow_sample['voxel_0']
        right_voxel = flow_sample['voxel_1']
        flow_gt = flow_sample['target']
        left_rgb = flow_sample['rgb_0']
        right_rgb = flow_sample['rgb_1']

        if isinstance(left_voxel, torch.Tensor):
            left_voxel = left_voxel.cpu().numpy()
        if isinstance(right_voxel, torch.Tensor):
            right_voxel = right_voxel.cpu().numpy()
        if isinstance(flow_gt, torch.Tensor):
            flow_gt = flow_gt.cpu().numpy()
        if isinstance(left_rgb, torch.Tensor):
            left_rgb = left_rgb.cpu().numpy()
        if isinstance(right_rgb, torch.Tensor):
            right_rgb = right_rgb.cpu().numpy()

        left_hist = self._voxel_to_2ch(left_voxel)
        right_hist = self._voxel_to_2ch(right_voxel)

        if self.use_start_no:
            depth = self.depth[self.start_no + idx]
        else:
            depth = self.depth[idx]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth, 0.01, 200.0)
        if depth.shape != (self.h, self.w):
            import cv2
            depth = cv2.resize(depth, (self.w, self.h))

        if 'indoor' in self.sequence:
            max_depth = 20.0
        else:
            max_depth = 200.0
        depth_normalized = np.log1p(depth) / np.log1p(max_depth)

        max_flow = 20.0
        flow = np.clip(flow_gt, -max_flow, max_flow) / max_flow
        if flow.shape[1:] != (self.h, self.w):
            import cv2
            flow = cv2.resize(flow.transpose(1,2,0), (self.w, self.h)).transpose(2,0,1)

        left_event = torch.from_numpy(left_hist).float()
        right_event = torch.from_numpy(right_hist).float()
        disp_left = torch.from_numpy(depth_normalized).float().unsqueeze(0)
        flow_gt = torch.from_numpy(flow).float()
        depth_raw = torch.from_numpy(depth).float().unsqueeze(0)

        left_rgb = torch.from_numpy(left_rgb).float() / 255.0
        right_rgb = torch.from_numpy(right_rgb).float() / 255.0

        return left_event, right_event, disp_left, flow_gt, left_rgb, right_rgb, depth_raw, max_depth

    def _voxel_to_2ch(self, voxel):
        pos = np.clip(voxel, 0, None).sum(axis=0, keepdims=True)
        neg = -np.clip(voxel, None, 0).sum(axis=0, keepdims=True)
        hist = np.concatenate([pos, neg], axis=0)
        max_val = hist.max()
        if max_val > 0:
            hist = hist / max_val
        return hist.astype(np.float32)


def main():


    DATA_HDF5_DIR = os.path.join(PROJECT_ROOT, 'Data', 'data_hdf5')
    _seq_names = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3', 'outdoor_day1']
    data_paths = {s: os.path.join(DATA_HDF5_DIR, f'{s}_data.hdf5') for s in _seq_names}
    gt_paths = {s: os.path.join(DATA_HDF5_DIR, f'{s}_gt.hdf5') for s in _seq_names}


    pretrained_checkpoint = os.path.join(
        PROJECT_ROOT, 'checkpoints', 'pretrained', 'unified', 'mvsec_out2_dt1', 'default', 'checkpoint_latest.pth')

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_checkpoint', type=str, default=pretrained_checkpoint)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_stage1', type=float, default=1e-4)
    parser.add_argument('--lr_stage2', type=float, default=1e-5)
    parser.add_argument('--lr_depth_head', type=float, default=1e-4)
    parser.add_argument('--num_epochs_stage1', type=int, default=10)
    parser.add_argument('--num_epochs_stage2', type=int, default=200)
    parser.add_argument('--flow_loss_weight', type=float, default=3.0)
    parser.add_argument('--depth_loss_weight', type=float, default=5.0)
    parser.add_argument('--depth_smooth_weight', type=float, default=0.1)
    parser.add_argument('--ms_gamma', type=float, default=0.8)
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(PROJECT_ROOT, 'checkpoints_joint_flow_depth'))
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.info('Loading all datasets with RGB...')
    sequences = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3', 'outdoor_day1']
    datasets = []
    for seq in sequences:
        dataset = MVSECWithDepthDatasetRGB(data_paths[seq], gt_paths[seq], sequence=seq)
        datasets.append(dataset)
        logging.info(f'  - {seq}: {len(dataset)} samples')

    combined_dataset = ConcatDataset(datasets)
    total_samples = len(combined_dataset)
    logging.info(f'Total combined samples: {total_samples}')

    weights = []
    dataset_sizes = [len(d) for d in datasets]
    avg_size = sum(dataset_sizes) / len(dataset_sizes)
    for i, ds in enumerate(datasets):
        weight = avg_size / dataset_sizes[i]
        weights.extend([weight] * len(ds))

    sampler = torch.utils.data.WeightedRandomSampler(weights, total_samples, replacement=True)

    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logging.info('Initializing model...')
    class Config:
        def __init__(self):
            self.no_batch_norm = False
            self.use_rgb = True

    model = EventDepthFlowNet(Config())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f'Using device: {device}')

    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        logging.info(f'Loading pretrained model from {args.pretrained_checkpoint}')
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        state_dict = checkpoint.get('model', checkpoint)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        new_state_dict = {k[7:]: v for k, v in state_dict.items()} if 'module.' in list(state_dict.keys())[0] else state_dict
        model.load_state_dict(new_state_dict, strict=False)
        logging.info('Pretrained model loaded successfully!')

    def compute_depth_smooth_loss(depth_pred):
        dx = torch.abs(depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:])
        dy = torch.abs(depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :])
        return torch.mean(dx) + torch.mean(dy)

    def depth_mae_loss(pred_norm, gt_norm, max_depth):
        log_max_depth = torch.log1p(max_depth)
        pred_depth = (torch.exp(pred_norm * log_max_depth) - 1).clamp(0.01, max_depth)
        gt_depth = (torch.exp(gt_norm * log_max_depth) - 1).clamp(0.01, max_depth)
        return torch.mean(torch.abs(pred_depth - gt_depth))

    def flow_criterion(flow_pred, flow_gt):
        diff = flow_pred - flow_gt
        epe_squared = torch.sum(diff ** 2, dim=1, keepdim=True)
        epe = torch.sqrt(epe_squared + 1e-8)
        robust_loss = torch.where(epe < 1.0, 0.5 * epe_squared, epe - 0.5)
        return torch.mean(robust_loss)

    def multi_scale_losses(depth_preds, flow_preds, disp_left_gt, flow_gt, max_depth_val, gamma):

        ms_depth_loss = 0.0
        ms_flow_loss = 0.0
        K = len(depth_preds)
        for k in range(K):
            scale_weight = gamma ** (K - 1 - k)
            pred_d = depth_preds[k]
            pred_f = flow_preds[k]

            gt_d = F.interpolate(disp_left_gt, size=pred_d.shape[-2:], mode='bilinear', align_corners=True)
            gt_f = F.interpolate(flow_gt, size=pred_f.shape[-2:], mode='bilinear', align_corners=True)

            ms_depth_loss = ms_depth_loss + scale_weight * depth_mae_loss(pred_d, gt_d, max_depth_val)
            ms_flow_loss = ms_flow_loss + scale_weight * flow_criterion(pred_f, gt_f)
        return ms_depth_loss, ms_flow_loss

    logging.info('=' * 60)
    logging.info('STAGE 1: Training RGB encoder and depth head')
    logging.info('=' * 60)

    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.rgb_encoder.named_parameters():
        param.requires_grad = True

    for name, param in model.depth_output.named_parameters():
        param.requires_grad = True

    for name, param in model.flow_output.named_parameters():
        param.requires_grad = True

    optimizer_stage1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_stage1, weight_decay=1e-5)
    scaler = GradScaler(enabled=args.use_amp and torch.cuda.is_available())

    best_loss = float('inf')
    for epoch in range(args.num_epochs_stage1):
        model.train()
        running_loss = 0.0
        running_epe = 0.0
        running_mae = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Stage1 Epoch {epoch+1}/{args.num_epochs_stage1}')
        for batch_idx, (left_event, right_event, disp_left_gt, flow_gt, left_rgb, right_rgb, depth_raw, max_depth) in pbar:
            left_event = left_event.to(device)
            right_event = right_event.to(device)
            disp_left_gt = disp_left_gt.to(device)
            flow_gt = flow_gt.to(device)
            left_rgb = left_rgb.to(device)
            right_rgb = right_rgb.to(device)
            max_depth = max_depth.to(device)

            optimizer_stage1.zero_grad()
            with autocast(enabled=args.use_amp and torch.cuda.is_available()):
                outputs = model(left_event, right_event, left_rgb, right_rgb)
                depth_pred = outputs['depth']
                flow_pred = outputs['flow']
                depth_preds = outputs['depth_preds']
                flow_preds = outputs['flow_preds']


                ms_depth_loss, ms_flow_loss = multi_scale_losses(
                    depth_preds, flow_preds, disp_left_gt, flow_gt, max_depth[0], args.ms_gamma)

                depth_loss = args.depth_loss_weight * ms_depth_loss
                flow_loss = args.flow_loss_weight * ms_flow_loss
                smooth_loss = args.depth_smooth_weight * compute_depth_smooth_loss(depth_pred)
                total_loss = depth_loss + flow_loss + smooth_loss


            scaler.scale(total_loss).backward()
            scaler.step(optimizer_stage1)
            scaler.update()

            running_loss += total_loss.item()


            diff = flow_pred - flow_gt
            epe = torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-8)
            running_epe += torch.mean(epe).item() * 20.0

            pred_depth_metric = (torch.exp(depth_pred * torch.log1p(max_depth[0])) - 1).clamp(0.01, max_depth[0])
            gt_depth_metric = (torch.exp(disp_left_gt * torch.log1p(max_depth[0])) - 1).clamp(0.01, max_depth[0])
            running_mae += torch.mean(torch.abs(pred_depth_metric - gt_depth_metric)).item()

            if (batch_idx + 1) % (len(dataloader) // 10) == 0:
                pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})

        avg_loss = running_loss / len(dataloader)
        avg_epe = running_epe / len(dataloader)
        avg_mae = running_mae / len(dataloader)

        logging.info(f'Stage1 Epoch {epoch+1}: Loss={avg_loss:.4f}, EPE={avg_epe:.4f}px, MAE={avg_mae:.4f}m')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'stage1_best.pth'))

    logging.info('=' * 60)
    logging.info('STAGE 2: Fine-tuning all with depth-focused learning')
    logging.info('=' * 60)

    for name, param in model.named_parameters():
        param.requires_grad = True

    backbone_params = []
    rgb_params = []
    depth_head_params = []
    flow_head_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'rgb_encoder' in name:
                rgb_params.append(param)
            elif 'depth_output' in name:
                depth_head_params.append(param)
            elif 'flow_output' in name:
                flow_head_params.append(param)
            elif 'encoder' in name.lower():
                backbone_params.append(param)
            else:
                other_params.append(param)

    optimizer_stage2 = optim.Adam([
        {'params': backbone_params, 'lr': args.lr_stage2 * 0.1},
        {'params': rgb_params, 'lr': args.lr_stage2},
        {'params': depth_head_params, 'lr': args.lr_depth_head},
        {'params': flow_head_params, 'lr': args.lr_stage2},
        {'params': other_params, 'lr': args.lr_stage2}
    ], weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=args.num_epochs_stage2, eta_min=1e-7)

    best_loss = float('inf')
    best_epe = float('inf')
    best_mae = float('inf')
    patience_counter = 0

    for epoch in range(args.num_epochs_stage2):
        model.train()
        running_loss = 0.0
        running_epe = 0.0
        running_mae = 0.0
        start_time = time.time()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Stage2 Epoch {epoch+1}/{args.num_epochs_stage2}')
        for batch_idx, (left_event, right_event, disp_left_gt, flow_gt, left_rgb, right_rgb, depth_raw, max_depth) in pbar:
            left_event = left_event.to(device)
            right_event = right_event.to(device)
            disp_left_gt = disp_left_gt.to(device)
            flow_gt = flow_gt.to(device)
            left_rgb = left_rgb.to(device)
            right_rgb = right_rgb.to(device)
            max_depth = max_depth.to(device)

            optimizer_stage2.zero_grad()
            with autocast(enabled=args.use_amp and torch.cuda.is_available()):
                outputs = model(left_event, right_event, left_rgb, right_rgb)
                depth_pred = outputs['depth']
                flow_pred = outputs['flow']
                depth_preds = outputs['depth_preds']
                flow_preds = outputs['flow_preds']


                ms_depth_loss, ms_flow_loss = multi_scale_losses(
                    depth_preds, flow_preds, disp_left_gt, flow_gt, max_depth[0], args.ms_gamma)

                depth_loss = args.depth_loss_weight * ms_depth_loss
                flow_loss = args.flow_loss_weight * ms_flow_loss
                smooth_loss = args.depth_smooth_weight * compute_depth_smooth_loss(depth_pred)
                total_loss = depth_loss + flow_loss + smooth_loss


            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer_stage2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer_stage2)
            scaler.update()

            running_loss += total_loss.item()


            diff = flow_pred - flow_gt
            epe = torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-8)
            running_epe += torch.mean(epe).item() * 20.0

            pred_depth_metric = (torch.exp(depth_pred * torch.log1p(max_depth[0])) - 1).clamp(0.01, max_depth[0])
            gt_depth_metric = (torch.exp(disp_left_gt * torch.log1p(max_depth[0])) - 1).clamp(0.01, max_depth[0])
            running_mae += torch.mean(torch.abs(pred_depth_metric - gt_depth_metric)).item()

            if (batch_idx + 1) % (len(dataloader) // 10) == 0:
                pbar.set_postfix({'Loss': f'{total_loss.item():.4f}', 'MAE': f'{running_mae/(batch_idx+1):.4f}m'})

        avg_loss = running_loss / len(dataloader)
        avg_epe = running_epe / len(dataloader)
        avg_mae = running_mae / len(dataloader)

        scheduler.step()

        logging.info(f'Stage2 Epoch {epoch+1}: Loss={avg_loss:.4f}, EPE={avg_epe:.4f}px, MAE={avg_mae:.4f}m, Time={time.time()-start_time:.2f}s')

        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'model_stage2_epoch_{epoch+1}.pth'))

        if avg_mae < best_mae:
            best_loss = avg_loss
            best_epe = avg_epe
            best_mae = avg_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model_rgb.pth'))
            logging.info(f'New best model saved! EPE: {best_epe:.4f}, MAE: {best_mae:.4f}m')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logging.info(f'Early stopping at epoch {epoch+1}, patience exceeded')
            break

    logging.info('Training completed!')
    logging.info(f'Best metrics: EPE={best_epe:.4f}px, MAE={best_mae:.4f}m')


if __name__ == '__main__':
    main()
