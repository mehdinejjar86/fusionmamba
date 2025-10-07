import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import List, Union, Tuple
from utils import read_image, InputPadder


class VFIFusionDataset(Dataset):
    """
    Multi-path dataset for VFI Fusion training
    Loads anchor frame pairs (I0, I1) and ground truth without precomputing flows
    """
    def __init__(self, gt_paths, steps, num_anchors=3, scale=1.0, 
                 mix_strategy='uniform', path_weights=None, divisor=32):
        """
        Args:
            gt_paths (str or list): Single path or list of paths to ground truth images
            steps (int or list): Single step or list of steps for each path
            num_anchors (int): Number of anchor frames (N in the model)
            scale (float): Scale factor for processing (1.0 recommended)
            mix_strategy (str): How to mix different paths ('uniform', 'weighted', 'sequential', 'balanced')
            path_weights (list): Weights for each path (used with 'weighted' strategy)
            divisor (int): Padding divisor (32 for VFI models)
        """
        # Handle single or multiple paths
        if isinstance(gt_paths, (str, Path)):
            self.gt_paths = [Path(gt_paths)]
        else:
            self.gt_paths = [Path(p) for p in gt_paths]
        
        # Handle single or multiple steps
        if isinstance(steps, int):
            self.steps = [steps] * len(self.gt_paths)
        else:
            if len(steps) != len(self.gt_paths):
                raise ValueError(f"Number of steps ({len(steps)}) must match number of paths ({len(self.gt_paths)})")
            self.steps = steps
        
        self.num_anchors = num_anchors
        self.scale = scale
        self.mix_strategy = mix_strategy
        self.divisor = divisor
        
        assert self.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
        
        # Image properties (set on first load)
        self.frame_dtype = None
        self.max_val = None
        self.h = None
        self.w = None
        self.padder = None
        
        # Setup path weights for weighted sampling
        if path_weights is not None:
            if len(path_weights) != len(self.gt_paths):
                raise ValueError("path_weights must match number of paths")
            self.path_weights = np.array(path_weights) / np.sum(path_weights)
        else:
            self.path_weights = np.ones(len(self.gt_paths)) / len(self.gt_paths)
        
        # Generate sequences for all paths
        self.all_samples = []
        self.path_sample_indices = []
        
        print(f"\nProcessing {len(self.gt_paths)} dataset paths:")
        print("-" * 50)
        
        for path_idx, (gt_path, step) in enumerate(zip(self.gt_paths, self.steps)):
            print(f"\nPath {path_idx + 1}: {gt_path}")
            print(f"  Step size: {step}")
            
            samples = self._process_single_path(gt_path, step, path_idx)
            start_idx = len(self.all_samples)
            self.all_samples.extend(samples)
            end_idx = len(self.all_samples)
            
            self.path_sample_indices.append((start_idx, end_idx))
            print(f"  Generated {len(samples)} samples")
        
        print("-" * 50)
        print(f"Total samples across all paths: {len(self.all_samples)}")
        
        # Create sample order based on mix strategy
        self._create_sample_order()
        

    
    def _setup_image_properties(self, gt_path, frame_num):
        """Setup image properties based on first frame"""
        img_path = gt_path / f"{frame_num}.png"
        if not img_path.exists():
            img_path = gt_path / f"{frame_num}.jpg"
        
        frame = read_image(str(img_path), img_path.suffix)
        self.frame_dtype = frame.dtype
        
        if self.frame_dtype == np.uint8:
            self.max_val = 255.
        elif self.frame_dtype == np.uint16:
            self.max_val = 65535.
        else:
            self.max_val = 1.
        
        h, w, _ = frame.shape
        self.h, self.w = h, w
        
        # Create InputPadder with the specified divisor
        self.padder = InputPadder((h, w), divisor=self.divisor)
        print(f"  Padding: {self.padder._pad} (divisor={self.divisor})")
    
    def _process_single_path(self, gt_path: Path, step: int, path_idx: int) -> List[dict]:
        """Process a single dataset path and generate samples"""
        # Find all image files
        gt_files = list(gt_path.glob("*.png"))
        if not gt_files:
            gt_files = list(gt_path.glob("*.jpg"))
        if not gt_files:
            raise ValueError(f"No image files found in {gt_path}")
        
        # Extract frame numbers
        gt_frames = []
        for file in gt_files:
            try:
                frame_num = int(file.stem)
                gt_frames.append(frame_num)
            except ValueError:
                continue
        
        gt_frames.sort()
        print(f"  Found {len(gt_frames)} frames: {min(gt_frames)} to {max(gt_frames)}")
        
        # Generate input sequence (frames at step intervals)
        input_frames = []
        current_frame = gt_frames[0]
        max_frame = gt_frames[-1]
        
        while current_frame <= max_frame:
            if current_frame in gt_frames:
                input_frames.append(current_frame)
            current_frame += step
        
        # Setup image properties if first path
        if path_idx == 0 and self.frame_dtype is None:
            self._setup_image_properties(gt_path, input_frames[0])
            print(f"  Image properties: {self.h}x{self.w}, dtype: {self.frame_dtype}")
        
        # Generate valid samples for this path
        samples = []
        
        for i in range(len(input_frames) - 1):
            # Skip if we don't have enough anchor frames on either side
            if i < self.num_anchors - 1 or i >= len(input_frames) - self.num_anchors:
                continue
            
            # Check for intermediate frames to interpolate
            for frame_num in range(input_frames[i] + 1, input_frames[i + 1]):
                if frame_num not in gt_frames:
                    continue
                
                # Collect anchor information
                anchor_info = []
                for anchor_idx in reversed(range(self.num_anchors)):
                    I0_index = i - anchor_idx
                    I1_index = i + anchor_idx + 1
                    
                    # Calculate timestep for this anchor
                    timestep = (frame_num - input_frames[I0_index]) / \
                              (input_frames[I1_index] - input_frames[I0_index])
                    
                    anchor_info.append({
                        'I0_frame': input_frames[I0_index],
                        'I1_frame': input_frames[I1_index],
                        'timestep': timestep
                    })
                
                samples.append({
                    'target_frame': frame_num,
                    'input_index': i,
                    'anchor_info': anchor_info,
                    'path_idx': path_idx,
                    'gt_path': gt_path,
                    'step': step
                })
        
        return samples
    
    def _create_sample_order(self):
        """Create sample ordering based on mix strategy"""
        n_samples = len(self.all_samples)
        
        if self.mix_strategy == 'sequential':
            self.sample_order = list(range(n_samples))
        
        elif self.mix_strategy == 'uniform':
            self.sample_order = list(range(n_samples))
            random.shuffle(self.sample_order)
        
        elif self.mix_strategy == 'weighted':
            self.sample_order = []
            for _ in range(n_samples):
                path_idx = np.random.choice(len(self.gt_paths), p=self.path_weights)
                start_idx, end_idx = self.path_sample_indices[path_idx]
                
                if start_idx < end_idx:
                    sample_idx = np.random.randint(start_idx, end_idx)
                    self.sample_order.append(sample_idx)
        
        elif self.mix_strategy == 'balanced':
            self.sample_order = []
            path_iterators = []
            
            for start_idx, end_idx in self.path_sample_indices:
                indices = list(range(start_idx, end_idx))
                random.shuffle(indices)
                path_iterators.append(iter(indices))
            
            # Round-robin sampling from each path
            path_idx = 0
            exhausted_paths = set()
            
            while len(exhausted_paths) < len(self.gt_paths):
                if path_idx not in exhausted_paths:
                    try:
                        sample_idx = next(path_iterators[path_idx])
                        self.sample_order.append(sample_idx)
                    except StopIteration:
                        exhausted_paths.add(path_idx)
                
                path_idx = (path_idx + 1) % len(self.gt_paths)
    
    def shuffle(self):
        """Reshuffle the dataset (call between epochs)"""
        if self.mix_strategy in ['uniform', 'weighted', 'balanced']:
            self._create_sample_order()
    
    def _load_and_process_image(self, gt_path: Path, frame_num: int):
        """Load and process a single image to tensor"""
        img_path = gt_path / f"{frame_num}.png"
        if not img_path.exists():
            img_path = gt_path / f"{frame_num}.jpg"
        
        img = read_image(str(img_path), img_path.suffix)
        img_tensor = torch.from_numpy(np.transpose(img.astype(np.int64), (2, 0, 1)))
        img_tensor = img_tensor.float() / self.max_val
        
        # Apply InputPadder
        img_tensor_padded = self.padder.pad(img_tensor)[0]
        
        return img_tensor_padded
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - I0: [N, 3, H', W'] - anchor start frames (padded)
                - I1: [N, 3, H', W'] - anchor end frames (padded)
                - timesteps: [N] - timesteps for each anchor
                - I_gt: [3, H', W'] - ground truth interpolated frame (padded)
                - target_frame: int - frame number
                - path_idx: int - which dataset path
        """
        # Get actual sample index based on mix strategy
        if hasattr(self, 'sample_order'):
            actual_idx = self.sample_order[idx % len(self.sample_order)]
        else:
            actual_idx = idx
        
        sample_info = self.all_samples[actual_idx]
        target_frame = sample_info['target_frame']
        anchor_info = sample_info['anchor_info']
        gt_path = sample_info['gt_path']
        path_idx = sample_info['path_idx']
        
        # Setup image properties on first access
        if self.frame_dtype is None:
            self._setup_image_properties(gt_path, target_frame)
        
        # Collect anchor frames and timesteps
        I0_list = []
        I1_list = []
        timesteps = []
        
        for anchor_data in anchor_info:
            I0_frame = anchor_data['I0_frame']
            I1_frame = anchor_data['I1_frame']
            timestep = anchor_data['timestep']
            
            # Load images (already padded)
            I0 = self._load_and_process_image(gt_path, I0_frame)
            I1 = self._load_and_process_image(gt_path, I1_frame)
            
            I0_list.append(I0)
            I1_list.append(I1)
            timesteps.append(timestep)
        
        # Load ground truth (already padded)
        I_gt = self._load_and_process_image(gt_path, target_frame)
        
        # Stack tensors
        I0_stacked = torch.stack(I0_list, dim=0)  # [N, 3, H', W']
        I1_stacked = torch.stack(I1_list, dim=0)  # [N, 3, H', W']
        timesteps_tensor = torch.tensor(timesteps, dtype=torch.float32)  # [N]
        
        return {
            'I0': I0_stacked,
            'I1': I1_stacked,
            'timesteps': timesteps_tensor,
            'I_gt': I_gt,
            'target_frame': target_frame,
            'path_idx': path_idx
        }
    
    def get_sample_info(self, idx):
        """Get human-readable information about a sample"""
        if hasattr(self, 'sample_order'):
            actual_idx = self.sample_order[idx % len(self.sample_order)]
        else:
            actual_idx = idx
        
        sample_info = self.all_samples[actual_idx]
        return {
            'target_frame': sample_info['target_frame'],
            'anchor_pairs': [(info['I0_frame'], info['I1_frame']) 
                           for info in sample_info['anchor_info']],
            'timesteps': [info['timestep'] for info in sample_info['anchor_info']],
            'path': str(sample_info['gt_path']),
            'step': sample_info['step'],
            'path_idx': sample_info['path_idx']
        }
    
    def get_path_statistics(self):
        """Get statistics about samples from each path"""
        stats = {}
        for idx, (path, step) in enumerate(zip(self.gt_paths, self.steps)):
            start_idx, end_idx = self.path_sample_indices[idx]
            stats[str(path)] = {
                'step': step,
                'num_samples': end_idx - start_idx,
                'percentage': (end_idx - start_idx) / len(self.all_samples) * 100
            }
        return stats


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    batched = {}
    for key in batch[0].keys():
        if key in ['target_frame', 'path_idx']:
            batched[key] = [sample[key] for sample in batch]
        else:
            batched[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return batched


def create_fusion_dataloader(gt_paths, steps, num_anchors=3, scale=1.0,
                             batch_size=4, shuffle=True, num_workers=4,
                             mix_strategy='uniform', path_weights=None, divisor=32):
    """
    Create a DataLoader for VFI Fusion training
    
    Args:
        divisor: Padding divisor (32 for VFI models, 64 for higher res)
    
    Returns:
        dataloader, dataset
    """
    dataset = VFIFusionDataset(
        gt_paths=gt_paths,
        steps=steps,
        num_anchors=num_anchors,
        scale=scale,
        mix_strategy=mix_strategy,
        path_weights=path_weights,
        divisor=divisor
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    return dataloader, dataset


def main():
    """Example usage and testing"""
    parser = argparse.ArgumentParser(description='VFI Fusion Dataset')
    parser.add_argument('--gt_paths', type=str, nargs='+', required=True,
                        help='Paths to ground truth images')
    parser.add_argument('--steps', type=int, nargs='+', required=True,
                        help='Step sizes for each path')
    parser.add_argument('--num_anchors', type=int, default=3,
                        help='Number of anchor frames (N)')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--divisor', type=int, default=32,
                        help='Padding divisor')
    parser.add_argument('--mix_strategy', type=str, default='uniform',
                        choices=['uniform', 'weighted', 'sequential', 'balanced'])
    parser.add_argument('--path_weights', type=float, nargs='+')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    print("Creating VFI Fusion dataset...")
    dataloader, dataset = create_fusion_dataloader(
        gt_paths=args.gt_paths,
        steps=args.steps,
        num_anchors=args.num_anchors,
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mix_strategy=args.mix_strategy,
        path_weights=args.path_weights,
        divisor=args.divisor
    )
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    stats = dataset.get_path_statistics()
    for path, info in stats.items():
        print(f"Path: {path}")
        print(f"  Step: {info['step']}, Samples: {info['num_samples']} ({info['percentage']:.1f}%)")
    print("-" * 50)
    print(f"Total: {len(dataset)} samples")
    
    # Test loading
    print(f"\nTesting {args.num_workers} workers...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:
            break
        print(f"\nBatch {batch_idx}:")
        print(f"  I0: {batch['I0'].shape}")  # [B, N, 3, H', W']
        print(f"  I1: {batch['I1'].shape}")  # [B, N, 3, H', W']
        print(f"  timesteps: {batch['timesteps'].shape}")  # [B, N]
        print(f"  I_gt: {batch['I_gt'].shape}")  # [B, 3, H', W']
        print(f"  Example timesteps: {batch['timesteps'][0].tolist()}")
    
    print("\n✓ Dataset ready for training!")


if __name__ == "__main__":
    main()