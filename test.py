import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SEQS = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3']


class DatasetRGB(torch.utils.data.Dataset):


    def __init__(self, data_path, gt_path, sequence, dt=1, bins=15, crop_size=(240, 320)):
        from datasets.MVSEC.MVSECProvider import MVSECProvider
        self.h, self.w = crop_size
        self.sequence = sequence
        data_root = os.path.dirname(os.path.dirname(data_path))
        provider = MVSECProvider({'name': 'mvsec', 'task': 'flow',
            'SequenceClass': 'MVSECSequence_flow_with_image',
            'SequenceNames': [sequence], 'root_path': data_root, 'split': 'test',
            'location': 'left', 'dt': dt, 'voxel_bins': bins,
            'crop_size': [self.h, self.w], 'augment': False})
        self.mvsec_seqs, _ = provider.get_datasets(F_list=True)
        self.flow_ds = provider.get_datasets()
        self.start_no = self.mvsec_seqs[0].start_no
        gt_file = h5py.File(gt_path, 'r')
        depth = gt_file['davis/left/depth_image_raw'][:]
        gt_file.close()
        total_depth = len(depth)
        avail = total_depth - self.start_no
        if avail > 0 and self.start_no < total_depth:
            self.length = min(len(self.flow_ds), avail)
            self.use_start_no = True
        else:
            self.length = min(len(self.flow_ds), total_depth)
            self.start_no = 0
            self.use_start_no = False
        self._depth_cache = depth
        print(f'  {sequence}: {self.length} samples')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s = self.flow_ds[idx]

        def to_np(x):
            return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        lv, rv, fg, lr, rr = map(to_np, [s['voxel_0'], s['voxel_1'], s['target'], s['rgb_0'], s['rgb_1']])

        def v2c(v):
            pos = np.clip(v, 0, None).sum(0, keepdims=True)
            neg = -np.clip(v, None, 0).sum(0, keepdims=True)
            h = np.concatenate([pos, neg], 0)
            mv = h.max()
            return (h / mv if mv > 0 else h).astype(np.float32)

        lh, rh = v2c(lv), v2c(rv)
        depth = self._depth_cache[self.start_no + idx] if self.use_start_no else self._depth_cache[idx]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.shape != (self.h, self.w):
            import cv2
            depth = cv2.resize(depth, (self.w, self.h))
        max_depth = 20.0 if 'indoor' in self.sequence else 200.0
        depth_raw = np.clip(depth, 0.01, max_depth)

        max_flow = 20.0
        flow = np.clip(fg, -max_flow, max_flow) / max_flow
        if flow.shape[1:] != (self.h, self.w):
            import cv2
            flow = cv2.resize(flow.transpose(1, 2, 0), (self.w, self.h)).transpose(2, 0, 1)

        return (torch.from_numpy(lh).float(), torch.from_numpy(rh).float(),
                torch.from_numpy(flow).float(),
                torch.from_numpy(lr).float() / 255.0, torch.from_numpy(rr).float() / 255.0,
                torch.from_numpy(depth_raw).float().unsqueeze(0),
                torch.tensor(max_depth, dtype=torch.float32))


def flow_metrics(fp, fg):

    diff = fp * 20.0 - fg * 20.0
    epe = torch.sqrt((diff ** 2).sum(1) + 1e-8)
    N = epe.numel()
    return (epe.mean().item(),
            (epe < 1).sum().item() / N * 100,
            (epe < 2).sum().item() / N * 100,
            (epe < 3).sum().item() / N * 100,
            (epe > 3).sum().item() / N * 100)


def depth_metrics(pred_norm, max_depth_val, depth_raw_m, min_valid=0.1):

    log_max = torch.log1p(max_depth_val)
    pred_m = torch.expm1(pred_norm * log_max).clamp(0.01, max_depth_val)
    valid = (depth_raw_m > min_valid) & torch.isfinite(depth_raw_m)
    ad = torch.abs(pred_m[valid] - depth_raw_m[valid])
    rel = ad / torch.clamp(depth_raw_m[valid], min=1e-4)
    return (ad.mean().item() * 100,
            torch.sqrt((ad ** 2).mean()).item() * 100,
            rel.mean().item(),
            (rel > 0.25).float().mean().item() * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_root', type=str, default=SCRIPT_DIR,
                        help='project root (must contain datasets/ and EventDepthFlowNet.py)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint path, default <project_root>/checkpoints_joint_flow_depth/best_model_rgb.pth')
    parser.add_argument('--seqs', type=str, nargs='+', default=DEFAULT_SEQS,
                        help='sequences to evaluate, default indoor_flying1/2/3')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='max batches per sequence, 0 means full evaluation')
    parser.add_argument('--out_json', type=str, default='test_joint_flow_depth_metrics.json')
    parser.add_argument('--out_txt', type=str, default='test_joint_flow_depth_report.txt')
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    data_dir = os.path.join(project_root, 'Data', 'data_hdf5')
    data_paths = {s: os.path.join(data_dir, f'{s}_data.hdf5') for s in args.seqs}
    gt_paths = {s: os.path.join(data_dir, f'{s}_gt.hdf5') for s in args.seqs}
    ckpt_path = args.ckpt or os.path.join(project_root, 'checkpoints_joint_flow_depth', 'best_model_rgb.pth')

    for p in [os.path.join(project_root, 'datasets', '__init__.py'),
              os.path.join(project_root, 'datasets', 'MVSEC', '__init__.py')]:
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'w').write('# marker\n')

    from EventDepthFlowNet import EventDepthFlowNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device={device}')

    class C:
        pass
    c = C()
    c.no_batch_norm = False
    c.use_rgb = True
    model = EventDepthFlowNet(c)

    if not os.path.exists(ckpt_path):
        print(f'[ERROR] checkpoint not found: {ckpt_path}')
        sys.exit(1)

    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    if len(sd) > 0 and list(sd.keys())[0].startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    msd = model.state_dict()
    ok = sum(1 for k, v in sd.items() if k in msd and v.shape == msd[k].shape)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    print(f'Loaded {os.path.basename(ckpt_path)} matched={ok}/{len(msd)} '
          f'(missing={len(missing)}, unexpected={len(unexpected)})')

    final = {}
    for seq in args.seqs:
        if not os.path.exists(data_paths[seq]) or not os.path.exists(gt_paths[seq]):
            print(f'[SKIP] {seq}: data/gt not found under {data_dir}')
            continue
        ds = DatasetRGB(data_paths[seq], gt_paths[seq], seq)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=False)
        ft, dt_tot, N = [0.0] * 5, [0.0] * 4, 0
        t0 = time.time()
        with torch.no_grad():
            for bi, (le, re, fgn, lr, rr, drm, md) in enumerate(tqdm(loader, leave=False, desc=seq)):
                if args.max_batches > 0 and bi >= args.max_batches:
                    break
                le, re, fgn = le.to(device), re.to(device), fgn.to(device)
                lr, rr, drm, md = lr.to(device), rr.to(device), drm.to(device), md.to(device)
                out = model(le, re, lr, rr)
                dp = out['depth'] if isinstance(out, dict) else out[0]
                fp = out['flow'] if isinstance(out, dict) else out[1]
                f = flow_metrics(fp, fgn)
                d = depth_metrics(dp, md[0], drm)
                for i in range(5):
                    ft[i] += f[i] * le.shape[0]
                for i in range(4):
                    dt_tot[i] += d[i] * le.shape[0]
                N += le.shape[0]
        if N == 0:
            continue
        flow = [x / N for x in ft]
        dep = [x / N for x in dt_tot]
        final[seq] = {
            'flow': {'EPE': flow[0], '1pe': flow[1], '2pe': flow[2], '3pe': flow[3], 'Outlier': flow[4]},
            'depth': {'MAE_cm': dep[0], 'RMSE_cm': dep[1], 'AbsRel': dep[2], 'D1': dep[3]},
            'num_samples': N,
            'elapsed_s': round(time.time() - t0, 1),
        }
        print(f'{seq} ({time.time()-t0:.0f}s, {N} samples): EPE={flow[0]:.4f} 1pe={flow[1]:.2f}% '
              f'2pe={flow[2]:.2f}% Out={flow[4]:.2f}% | MAE={dep[0]:.2f}cm RMSE={dep[1]:.2f}cm '
              f'AbsRel={dep[2]:.4f} D1={dep[3]:.2f}%')

    if not final:
        print('No sequence evaluated.')
        return

    ns = len(final)
    avg_flow = {k: sum(v['flow'][k] for v in final.values()) / ns for k in ['EPE', '1pe', '2pe', '3pe', 'Outlier']}
    avg_depth = {k: sum(v['depth'][k] for v in final.values()) / ns for k in ['MAE_cm', 'RMSE_cm', 'AbsRel', 'D1']}
    final['AVERAGE'] = {'flow': avg_flow, 'depth': avg_depth}

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append('=' * 70)
    lines.append('Multi-Scale Loss model (joint_flow_depth) evaluation report')
    lines.append(f'checkpoint: {ckpt_path}')
    lines.append(f'sequences: {", ".join(args.seqs)}    batch_size={args.batch_size}    max_batches={args.max_batches or "ALL"}')
    lines.append('=' * 70)
    lines.append('')
    lines.append('Flow (EPE in px, lower is better; 1pe/2pe/3pe = % of pixels with error < 1/2/3 px, higher is better; Outlier>3px %, lower is better)')
    lines.append(f'{"seq":<16}{"EPE":>10}{"1pe":>10}{"2pe":>10}{"3pe":>10}{"Outlier":>10}')
    for s, v in final.items():
        fl = v['flow']
        lines.append(f'{s:<16}{fl["EPE"]:>10.4f}{fl["1pe"]:>10.2f}{fl["2pe"]:>10.2f}{fl["3pe"]:>10.2f}{fl["Outlier"]:>10.2f}')
    lines.append('')
    lines.append('Depth (MAE/RMSE in cm, lower is better; AbsRel lower is better; D1 = % of pixels with relative error > 0.25, lower is better)')
    lines.append(f'{"seq":<16}{"MAE":>10}{"RMSE":>10}{"AbsRel":>10}{"D1":>10}')
    for s, v in final.items():
        dp = v['depth']
        lines.append(f'{s:<16}{dp["MAE_cm"]:>10.2f}{dp["RMSE_cm"]:>10.2f}{dp["AbsRel"]:>10.4f}{dp["D1"]:>10.2f}')
    lines.append('')
    lines.append(f'Saved JSON: {args.out_json}')

    report = '\n'.join(lines)
    print('\n' + report)
    with open(args.out_txt, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f'Saved report: {args.out_txt}')


if __name__ == '__main__':
    main()
