"""
T08: Trích đặc trưng từng frame bằng MobileNetV2 (Pretrained ImageNet)
Phụ trách: Nguyễn Xuân Quy (Mã B)
Đầu vào từ Hiếu (Mã A) tại `c:\\Lập Trình Trí Tuệ Nhân Tạo\\violence`:
  - `outputs/splits.csv`: 1986 video RLVS (train 1391, val 298, test 297)
  - `outputs/cross_test.csv`: 400 video RWF-2000 (200 Fight, 200 NonFight)
  - `frames/n8`, `frames/n16`, `frames/n20`: các file `<video_id>.npy` uint8 RGB (T, 224, 224, 3)
  - `preprocess_utils.py`: `load_frames`, `normalize`, `to_torch_layout`, `augment_clip`

Đầu ra tại `c:\\Thị giác máy tính\\Công việc\\T08-T09\\features\\`:
  - `features_n{T}.npz`: Cache nén tốc độ cao cho 1986 video RLVS + 1391 bản tăng cường tập train (T14)
  - `features_cross_n{T}.npz`: Cache nén tốc độ cao cho 400 video RWF-2000 (T20)
  - `T{T}/<video_id>.npy` & `T{T}_aug/<video_id>_aug1.npy`: File đặc trưng (T, 1280) từng video
  - `features_T{T}.csv`, `features_n{T}.csv`, `features_cross_T{T}.csv`, `features_cross_n{T}.csv`,
    `features_T{T}_aug.csv`, `features_n{T}_aug.csv`, `features_all_T{T}.csv`
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# Thêm đường dẫn thư mục dữ liệu của Hiếu (Mã A) để import preprocess_utils
DEFAULT_VIOLENCE_DIR = Path(r"c:\Lập Trình Trí Tuệ Nhân Tạo\violence")
if DEFAULT_VIOLENCE_DIR.exists() and str(DEFAULT_VIOLENCE_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_VIOLENCE_DIR))

try:
    from preprocess_utils import augment_clip, load_frames, normalize, to_torch_layout
except ImportError:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_frames(path):
        """uint8 (T,224,224,3), RGB."""
        return np.load(path)

    def normalize(frames_uint8):
        """Chuẩn hóa ImageNet, trả về float32 (T,224,224,3)."""
        x = frames_uint8.astype(np.float32) / 255.0
        return (x - MEAN) / STD

    def to_torch_layout(frames_uint8):
        """float32 (T,3,224,224) cho PyTorch."""
        return normalize(frames_uint8).transpose(0, 3, 1, 2)

    def augment_clip(frames, rng=None):
        """Tăng cường một video (lật ngang, đổi độ sáng, cắt ngẫu nhiên). CHỈ dùng cho tập train."""
        rng = rng or random.Random()
        out = frames
        if rng.random() < 0.5:
            out = out[:, :, ::-1]
        out = np.clip(out.astype(np.float32) * rng.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        T, H, W, _ = out.shape
        s = rng.uniform(0.8, 1.0)
        ch, cw = int(H * s), int(W * s)
        y, x = rng.randint(0, H - ch), rng.randint(0, W - cw)
        out = np.stack([cv2.resize(np.ascontiguousarray(f[y:y + ch, x:x + cw]), (W, H)) for f in out])
        return out.astype(np.uint8)


# Chuẩn hóa theo chuẩn ImageNet (dùng chung cho cả lúc trích frame trực tiếp và Demo T22)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_feature_extractor(device: torch.device) -> nn.Module:
    """
    Khởi tạo MobileNetV2 pretrained trên ImageNet, bỏ lớp phân loại cuối.
    Đầu ra sau Global Average Pooling có kích thước (batch_size, 1280).
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    backbone = models.mobilenet_v2(weights=weights)
    extractor = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
    )
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
    return extractor.to(device)


def sample_frames_from_video(
    video_path: Path,
    num_frames: int = 16,
    image_size: int = 224,
) -> np.ndarray:
    """
    Hàm tiện ích trích N frame cách đều từ file video, resize 224x224 và chuẩn hóa ImageNet.
    Trả về: numpy array kích thước (T, 3, 224, 224) kiểu float32.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()
        if not raw_frames:
            raise RuntimeError(f"Video rỗng hoặc lỗi đọc frame: {video_path}")
        indices = np.linspace(0, len(raw_frames) - 1, num_frames, dtype=int)
        selected = [raw_frames[idx] for idx in indices]
    else:
        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        selected = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                if selected:
                    frame = selected[-1].copy()
                else:
                    cap.release()
                    raise RuntimeError(f"Lỗi giải mã frame tại vị trí {idx} của {video_path}")
            selected.append(frame)
        cap.release()

    rgb_frames = []
    for frame_bgr in selected:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA
        )
        rgb_frames.append(frame_resized)

    frames_uint8 = np.stack(rgb_frames, axis=0).astype(np.uint8)
    return to_torch_layout(frames_uint8).astype(np.float32)


def normalize_frame_array(frames: np.ndarray) -> torch.Tensor:
    """
    Chuyển mảng frame .npy do A giao (uint8 (T, 224, 224, 3)) hoặc mảng đã chuẩn hóa
    về đúng tensor kích thước (T, 3, 224, 224) trên PyTorch bằng `to_torch_layout`.
    """
    if frames.ndim != 4:
        raise ValueError(f"Mảng frame phải có 4 chiều, nhưng nhận được: {frames.shape}")

    if frames.dtype == np.uint8:
        return torch.from_numpy(np.ascontiguousarray(to_torch_layout(frames)))

    if frames.shape[-1] == 3 and frames.shape[1] != 3:
        if frames.max() > 10.0:
            return torch.from_numpy(np.ascontiguousarray(to_torch_layout(frames.astype(np.uint8))))
        frames = np.transpose(frames, (0, 3, 1, 2))

    frames = frames.astype(np.float32)
    if frames.max() > 10.0:
        frames = frames / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        frames = (frames - mean) / std

    return torch.from_numpy(np.ascontiguousarray(frames))


@torch.no_grad()
def extract_features_for_video_frames(
    extractor: nn.Module,
    frames_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Nhận mảng frame của 1 video (T, 3, 224, 224) hoặc (T, 224, 224, 3),
    đưa qua MobileNetV2 và trả về mảng đặc trưng (T, 1280) kiểu float32.
    """
    frames_tensor = normalize_frame_array(frames_np).to(device)
    features = extractor(frames_tensor)  # (T, 1280)
    features_np = features.cpu().numpy().astype(np.float32)
    assert features_np.ndim == 2 and features_np.shape[1] == 1280, (
        f"Kích thước đặc trưng sai: {features_np.shape}, kỳ vọng (T, 1280)"
    )
    return features_np


def _load_and_prepare_clip(
    npy_path: Path,
    num_frames: int,
    apply_augment: bool = False,
    aug_seed: Optional[int] = None,
    return_uint8: bool = False,
) -> np.ndarray:
    """
    Đọc file .npy uint8 bằng `load_frames` của Hiếu, áp dụng `augment_clip` nếu yêu cầu,
    và chuyển sang định dạng PyTorch `(T, 3, 224, 224)` float32 bằng `to_torch_layout`
    (hoặc trả về uint8 `(T, 224, 224, 3)` để chuẩn hóa trực tiếp trên GPU).
    """
    frames_uint8 = load_frames(npy_path)
    if len(frames_uint8) != num_frames:
        idx = np.linspace(0, len(frames_uint8) - 1, num_frames, dtype=int)
        frames_uint8 = frames_uint8[idx]
    if apply_augment:
        rng = random.Random(aug_seed) if aug_seed is not None else random.Random()
        frames_uint8 = augment_clip(frames_uint8, rng=rng)
    if return_uint8:
        return np.ascontiguousarray(frames_uint8, dtype=np.uint8)
    return to_torch_layout(frames_uint8).astype(np.float32)


@torch.no_grad()
def batch_extract_dataframe(
    df: pd.DataFrame,
    frames_dir: Path,
    features_subdir: Path,
    extractor: nn.Module,
    device: torch.device,
    num_frames: int,
    batch_videos: int = 32,
    apply_augment: bool = False,
    base_seed: int = 42,
    desc: str = "Extracting",
    num_workers: int = 8,
) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất đặc trưng theo lô (batch) tốc độ cao cho toàn bộ bảng `df`.
    Kết hợp đọc/tiền xử lý đa luồng (ThreadPoolExecutor) với suy luận GPU theo batch.
    Trả về:
      - `all_features`: float32 shape `(N, T, 1280)`
      - `feature_paths`: danh sách đường dẫn file `.npy` từng video
    """
    features_subdir.mkdir(parents=True, exist_ok=True)
    n_videos = len(df)
    all_features = np.zeros((n_videos, num_frames, 1280), dtype=np.float32)
    feature_paths: List[str] = [""] * n_videos

    to_compute_indices: List[int] = []
    to_compute_paths: List[Path] = []
    to_compute_out_paths: List[Path] = []
    to_compute_seeds: List[int] = []

    video_ids = df["video_id"].astype(str).tolist()
    for i, vid in enumerate(video_ids):
        suffix = "_aug1" if apply_augment else ""
        out_path = features_subdir / f"{vid}{suffix}.npy"
        feature_paths[i] = str(out_path)

        if out_path.exists():
            try:
                cached = np.load(out_path)
                if cached.shape == (num_frames, 1280) and not np.isnan(cached).any():
                    all_features[i] = cached.astype(np.float32)
                    continue
            except Exception:
                pass

        npy_path = frames_dir / f"{vid}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Thiếu file frame .npy: {npy_path}")

        to_compute_indices.append(i)
        to_compute_paths.append(npy_path)
        to_compute_out_paths.append(out_path)
        to_compute_seeds.append(base_seed + i)

    if not to_compute_indices:
        print(f"  [CACHE HIT] Đã có sẵn {n_videos}/{n_videos} file tại {features_subdir}")
        return all_features, feature_paths

    mean_gpu = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_gpu = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for start in tqdm(
            range(0, len(to_compute_indices), batch_videos),
            desc=desc,
            unit="batch",
        ):
            end = min(start + batch_videos, len(to_compute_indices))
            batch_idx = to_compute_indices[start:end]
            batch_npy = to_compute_paths[start:end]
            batch_out = to_compute_out_paths[start:end]
            batch_seed = to_compute_seeds[start:end]

            clips = list(
                pool.map(
                    lambda args: _load_and_prepare_clip(*args),
                    [
                        (p, num_frames, apply_augment, s, True)
                        for p, s in zip(batch_npy, batch_seed)
                    ],
                )
            )
            # Ghép thành (B * T, 224, 224, 3) uint8 -> chuyển lên GPU và chuẩn hóa ImageNet
            bsz = len(clips)
            stacked_clips = np.concatenate(clips, axis=0)  # (B * T, 224, 224, 3) uint8
            tensor_u8 = torch.from_numpy(stacked_clips).to(device, non_blocking=True)
            tensor_in = tensor_u8.permute(0, 3, 1, 2).contiguous().to(torch.float32).div_(255.0)
            del tensor_u8
            tensor_in.sub_(mean_gpu).div_(std_gpu)
            feats_flat = extractor(tensor_in)  # (B * T, 1280)
            feats_batch = (
                feats_flat.view(bsz, num_frames, 1280).cpu().numpy().astype(np.float32)
            )
            del tensor_in, feats_flat

            for idx_in_df, out_p, feat_arr in zip(batch_idx, batch_out, feats_batch):
                all_features[idx_in_df] = feat_arr
                np.save(out_p, feat_arr)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return all_features, feature_paths


def extract_for_frame_count(
    num_frames: int,
    violence_dir: Path,
    output_dir: Path,
    extractor: nn.Module,
    device: torch.device,
    batch_videos: int = 32,
    extract_cross: bool = True,
    extract_aug: bool = True,
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Thực hiện trích xuất đầy đủ cho cấu hình `T = num_frames`:
      1. 1986 video RLVS (`splits.csv`) -> `features/T{T}/*.npy`, `features_T{T}.csv`, `features_n{T}.csv`
      2. (Tuỳ chọn) 400 video RWF-2000 (`cross_test.csv`) -> `features_cross_T{T}.csv`, `features_cross_n{T}.npz`
      3. (Tuỳ chọn) 1391 video RLVS `train` tăng cường (`augment_clip`) -> `features_T{T}_aug.csv`, lưu kèm trong `features_n{T}.npz`
    """
    t0 = time.time()
    splits_path = violence_dir / "outputs" / "splits.csv"
    cross_path = violence_dir / "outputs" / "cross_test.csv"
    frames_dir = violence_dir / "frames" / f"n{num_frames}"

    if not splits_path.exists():
        raise FileNotFoundError(f"Không tìm thấy splits.csv tại: {splits_path}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục frame tại: {frames_dir}")

    df_rlvs = pd.read_csv(splits_path)
    df_rlvs["label"] = df_rlvs["label"].astype(int)

    features_subdir = output_dir / f"T{num_frames}"
    print(f"\n=== [T={num_frames}] 1/3: Trích đặc trưng RLVS ({len(df_rlvs)} video) từ {frames_dir} ===")
    rlvs_feats, rlvs_paths = batch_extract_dataframe(
        df=df_rlvs,
        frames_dir=frames_dir,
        features_subdir=features_subdir,
        extractor=extractor,
        device=device,
        num_frames=num_frames,
        batch_videos=batch_videos,
        apply_augment=False,
        base_seed=seed,
        desc=f"RLVS n{num_frames}",
    )
    df_rlvs_out = df_rlvs.copy()
    df_rlvs_out["feature_path"] = rlvs_paths
    df_rlvs_out["is_aug"] = 0

    csv_t = output_dir / f"features_T{num_frames}.csv"
    csv_n = output_dir / f"features_n{num_frames}.csv"
    df_rlvs_out.to_csv(csv_t, index=False)
    df_rlvs_out.to_csv(csv_n, index=False)

    stacked_npy = output_dir / f"all_features_T{num_frames}.npy"
    np.save(stacked_npy, rlvs_feats)

    saved_paths: Dict[str, Path] = {
        "rlvs_csv_T": csv_t,
        "rlvs_csv_n": csv_n,
        "rlvs_stacked_npy": stacked_npy,
    }

    # 2. Trích xuất RWF-2000 cross_test.csv
    if extract_cross and cross_path.exists():
        df_cross = pd.read_csv(cross_path)
        df_cross["label"] = df_cross["label"].astype(int)
        print(f"=== [T={num_frames}] 2/3: Trích đặc trưng RWF-2000 Cross-Test ({len(df_cross)} video) ===")
        cross_feats, cross_paths = batch_extract_dataframe(
            df=df_cross,
            frames_dir=frames_dir,
            features_subdir=features_subdir,
            extractor=extractor,
            device=device,
            num_frames=num_frames,
            batch_videos=batch_videos,
            apply_augment=False,
            base_seed=seed + 10000,
            desc=f"RWF-2000 n{num_frames}",
        )
        df_cross_out = df_cross.copy()
        df_cross_out["feature_path"] = cross_paths
        df_cross_out["is_aug"] = 0

        cross_csv_t = output_dir / f"features_cross_T{num_frames}.csv"
        cross_csv_n = output_dir / f"features_cross_n{num_frames}.csv"
        df_cross_out.to_csv(cross_csv_t, index=False)
        df_cross_out.to_csv(cross_csv_n, index=False)

        cross_npz = output_dir / f"features_cross_n{num_frames}.npz"
        np.savez(
            cross_npz,
            features=cross_feats,
            labels=df_cross_out["label"].to_numpy(dtype=np.int64),
            splits=df_cross_out["split"].astype(str).to_numpy(),
            video_ids=df_cross_out["video_id"].astype(str).to_numpy(),
        )

        # Lưu thêm bảng gộp RLVS + RWF-2000 (2386 dòng)
        common_cols = [
            "video_id", "path", "label", "label_name", "split",
            "frames", "fps", "width", "height", "duration", "feature_path", "is_aug"
        ]
        df_all = pd.concat(
            [df_rlvs_out[common_cols], df_cross_out[common_cols]],
            ignore_index=True,
        )
        all_csv = output_dir / f"features_all_T{num_frames}.csv"
        df_all.to_csv(all_csv, index=False)

        saved_paths["cross_csv_T"] = cross_csv_t
        saved_paths["cross_csv_n"] = cross_csv_n
        saved_paths["cross_npz"] = cross_npz
        saved_paths["all_csv"] = all_csv

    # 3. Trích xuất bản tăng cường (augment_clip) trên tập train RLVS cho T14
    npz_payload = {
        "features": rlvs_feats,
        "labels": df_rlvs_out["label"].to_numpy(dtype=np.int64),
        "splits": df_rlvs_out["split"].astype(str).to_numpy(),
        "video_ids": df_rlvs_out["video_id"].astype(str).to_numpy(),
    }

    if extract_aug:
        df_train = df_rlvs[df_rlvs["split"] == "train"].copy().reset_index(drop=True)
        aug_subdir = output_dir / f"T{num_frames}_aug"
        print(f"=== [T={num_frames}] 3/3: Trích đặc trưng Train Augmented ({len(df_train)} video) với augment_clip ===")
        aug_feats, aug_paths = batch_extract_dataframe(
            df=df_train,
            frames_dir=frames_dir,
            features_subdir=aug_subdir,
            extractor=extractor,
            device=device,
            num_frames=num_frames,
            batch_videos=batch_videos,
            apply_augment=True,
            base_seed=seed + 50000 + num_frames * 1000,
            desc=f"Train-Aug n{num_frames}",
        )
        df_train_aug = df_train.copy()
        df_train_aug["video_id"] = df_train_aug["video_id"].astype(str) + "_aug1"
        df_train_aug["feature_path"] = aug_paths
        df_train_aug["is_aug"] = 1

        df_rlvs_with_aug = pd.concat([df_rlvs_out, df_train_aug], ignore_index=True)
        aug_csv_t = output_dir / f"features_T{num_frames}_aug.csv"
        aug_csv_n = output_dir / f"features_n{num_frames}_aug.csv"
        df_rlvs_with_aug.to_csv(aug_csv_t, index=False)
        df_rlvs_with_aug.to_csv(aug_csv_n, index=False)

        npz_payload["train_aug_features"] = aug_feats
        npz_payload["train_aug_labels"] = df_train["label"].to_numpy(dtype=np.int64)
        npz_payload["train_aug_video_ids"] = df_train["video_id"].astype(str).to_numpy()
        saved_paths["aug_csv_T"] = aug_csv_t
        saved_paths["aug_csv_n"] = aug_csv_n

    rlvs_npz = output_dir / f"features_n{num_frames}.npz"
    np.savez(rlvs_npz, **npz_payload)
    saved_paths["rlvs_npz"] = rlvs_npz

    elapsed = time.time() - t0
    print(
        f"[HOÀN TẤT n{num_frames} trong {elapsed:.1f}s] "
        f"RLVS={rlvs_feats.shape}, NPZ={rlvs_npz}"
    )
    return saved_paths


def run_dry_run_pipeline(
    output_dir: Path,
    num_samples: int = 24,
    num_frames: int = 16,
    seed: int = 42,
) -> None:
    """
    Chế độ kiểm tra nhanh (--dry_run): tạo dữ liệu đặc trưng giả lập đúng chuẩn
    (N, T, 1280) và file splits.csv mẫu để kiểm tra toàn bộ luồng T08 -> T09.
    """
    rng = np.random.default_rng(seed)
    features_dir = output_dir / f"T{num_frames}"
    features_dir.mkdir(parents=True, exist_ok=True)

    splits_order = (
        ["train"] * int(num_samples * 0.7)
        + ["val"] * int(num_samples * 0.15)
        + ["test"] * (num_samples - int(num_samples * 0.7) - int(num_samples * 0.15))
    )

    rows = []
    all_features = []
    for i, split in enumerate(splits_order):
        label = i % 2
        video_id = f"{'V' if label == 1 else 'NV'}_{i + 1:04d}"
        base_signal = 0.25 if label == 1 else -0.25
        feat = rng.normal(loc=base_signal, scale=1.0, size=(num_frames, 1280)).astype(
            np.float32
        )
        feat_path = features_dir / f"{video_id}.npy"
        np.save(feat_path, feat)
        all_features.append(feat)
        rows.append(
            {
                "video_id": video_id,
                "label": label,
                "split": split,
                "feature_path": str(feat_path),
            }
        )

    stacked = np.stack(all_features, axis=0)
    assert stacked.shape == (num_samples, num_frames, 1280), (
        f"Kích thước tensor tổng sai: {stacked.shape}"
    )

    df = pd.DataFrame(rows)
    metadata_csv = output_dir / f"features_T{num_frames}.csv"
    df.to_csv(metadata_csv, index=False)
    print(f"[DRY-RUN THÀNH CÔNG] Đã lưu {num_samples} file đặc trưng tại: {features_dir}")
    print(f"[KIỂM TRA TENSOR] Kích thước toàn tập (N, T, 1280) = {stacked.shape}")
    print(f"[METADATA] Đã lưu danh sách đặc trưng tại: {metadata_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T08: Trích đặc trưng MobileNetV2 (1280-dim) cho RLVS & RWF-2000"
    )
    parser.add_argument(
        "--violence_dir",
        type=str,
        default=str(DEFAULT_VIOLENCE_DIR),
        help="Thư mục gốc dữ liệu của Hiếu (chứa outputs/ và frames/)",
    )
    parser.add_argument(
        "--splits_csv",
        type=str,
        default=None,
        help="Đường dẫn tới file splits.csv (mặc định: <violence_dir>/outputs/splits.csv)",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Thư mục chứa các file frame .npy (mặc định: <violence_dir>/frames/n<T>)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        choices=[8, 16, 20],
        help="Số lượng frame T mỗi video (mặc định 16; hỗ trợ 8/16/20 cho T10)",
    )
    parser.add_argument(
        "--all_frames",
        action="store_true",
        help="Trích xuất đồng thời cả 3 cấu hình n16, n8, n20",
    )
    parser.add_argument(
        "--no_cross",
        action="store_true",
        help="Bỏ qua không trích xuất tập cross_test.csv (RWF-2000)",
    )
    parser.add_argument(
        "--no_aug",
        action="store_true",
        help="Bỏ qua không trích xuất bản tăng cường tập train (augment_clip)",
    )
    parser.add_argument(
        "--batch_videos",
        type=int,
        default=32,
        help="Số lượng video xử lý trong 1 batch trên GPU/CPU",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="features",
        help="Thư mục đầu ra lưu cache đặc trưng MobileNetV2",
    )
    parser.add_argument(
        "--save_stacked",
        action="store_true",
        help="Cờ tương thích ngược (luôn lưu file .npz và all_features_T{T}.npy)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Chạy kiểm tra luồng và kích thước tensor bằng dữ liệu giả lập",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        run_dry_run_pipeline(output_dir=output_dir, num_frames=args.num_frames)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Đang khởi tạo MobileNetV2 (ImageNet1K_V1) trên thiết bị: {device}")
    extractor = build_feature_extractor(device)

    violence_dir = Path(args.violence_dir)
    frame_list = [16, 8, 20] if args.all_frames else [args.num_frames]

    for n_frames in frame_list:
        extract_for_frame_count(
            num_frames=n_frames,
            violence_dir=violence_dir,
            output_dir=output_dir,
            extractor=extractor,
            device=device,
            batch_videos=args.batch_videos,
            extract_cross=not args.no_cross,
            extract_aug=not args.no_aug,
        )


if __name__ == "__main__":
    main()
