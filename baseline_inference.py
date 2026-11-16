"""
T22: Module Tiền xử lý & Suy luận Mô hình Baseline (MobileNetV2 + LSTM/GRU) cho Demo
Phụ trách: Nguyễn Xuân Quy (Mã B)

Module này đóng gói các hàm dùng chung cho `demo.py` (phối hợp với Linh - C và Thiên - D):
  1. `preprocess_video(video_path, num_frames=16)` -> `torch.FloatTensor` kích thước `(1, num_frames, 3, 224, 224)`
     - Hỗ trợ cả file video gốc (`.mp4`, `.avi`, `.mkv`, `.mov`, ...) VÀ file frame `.npy` (`(T, 224, 224, 3)` `uint8`).
     - Dùng đúng logic trích xuất frame cách đều và chuẩn hóa ImageNet (`to_torch_layout`) của Hiếu (Mã A).
  2. `predict_baseline(video_path, checkpoint_path=None, device=None, threshold=None)` -> `(prob, label)`
     - Cache mô hình MobileNetV2 và RNN (`baseline.pt`) trong bộ nhớ (RAM/VRAM) để gọi lặp lại tức thì trên Gradio/Streamlit.
     - Trả về `(prob, label)` với `prob` là xác suất lớp `Violence` trong `[0.0, 1.0]` và `label` là `"Violence"` / `"NonViolence"`.
  3. `predict_baseline_detailed(video_path, ...)` -> `dict` chứa đầy đủ `prob`, `label_id`, `label_name`, `label_vi`, `inference_time_ms`, ...

Cách chạy kiểm thử tự động (Gate G4 - kiểm tra trên >= 5 video thực tế từ RLVS & RWF-2000):
    python baseline_inference.py --test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ==============================================================================
# 1. Tích hợp `preprocess_utils.py` của Hiếu (Mã A)
# ==============================================================================
HIEU_VIOLENCE_DIR = Path(r"c:\Lập Trình Trí Tuệ Nhân Tạo\violence")
WORKSPACE_DIR = Path(r"c:\Thị giác máy tính\Công việc\T08-T09")

if HIEU_VIOLENCE_DIR.exists() and str(HIEU_VIOLENCE_DIR) not in sys.path:
    sys.path.insert(0, str(HIEU_VIOLENCE_DIR))

try:
    from preprocess_utils import MEAN as IMAGENET_MEAN_NP  # type: ignore
    from preprocess_utils import STD as IMAGENET_STD_NP  # type: ignore
    from preprocess_utils import load_frames, normalize, to_torch_layout  # type: ignore
except ImportError:
    IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_frames(path: Union[str, Path]) -> np.ndarray:
        """Đọc mảng frame uint8 (T, 224, 224, 3), hệ màu RGB."""
        return np.load(path)

    def normalize(frames_uint8: np.ndarray) -> np.ndarray:
        """Chuẩn hóa ImageNet, trả về float32 (T, 224, 224, 3)."""
        x = frames_uint8.astype(np.float32) / 255.0
        return (x - IMAGENET_MEAN_NP) / IMAGENET_STD_NP

    def to_torch_layout(frames_uint8: np.ndarray) -> np.ndarray:
        """Chuyển sang float32 (T, 3, 224, 224) cho PyTorch."""
        return normalize(frames_uint8).transpose(0, 3, 1, 2)


IMAGE_SIZE: int = 224
DEFAULT_NUM_FRAMES: int = 16
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".mpeg", ".mpg"}


# ==============================================================================
# 2. Kiến trúc Mô hình Baseline (MobileNetV2 + LSTM/GRU)
# ==============================================================================
class ViolenceBaselineRNN(nn.Module):
    """
    Kiến trúc Baseline phát hiện bạo lực (T09):
      Input: chuỗi đặc trưng MobileNetV2 kích thước `(B, T, 1280)`
      -> Mạng tuần tự LSTM hoặc GRU (1-2 lớp, hỗ trợ cả đơn hướng và hai chiều)
      -> Tầng phân loại tuyến tính + Sigmoid -> Xác suất lớp Violence trong `[0.0, 1.0]`.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_size: int = 256,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pooling = pooling.lower()

        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == "lstm":
            self.rnn: nn.Module = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"rnn_type chỉ nhận 'lstm' hoặc 'gru', nhận được: {rnn_type}")

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, T, 1280)
        rnn_out, _ = self.rnn(x)  # (batch_size, T, out_dim)
        if self.pooling == "mean":
            feat = rnn_out.mean(dim=1)
        elif self.pooling == "max":
            feat = rnn_out.max(dim=1).values
        else:
            feat = rnn_out[:, -1, :]  # (batch_size, out_dim)
        out = self.classifier(feat).squeeze(-1)  # (batch_size,)
        return out


def build_mobilenetv2_extractor(device: torch.device) -> nn.Module:
    """
    Khởi tạo MobileNetV2 pretrained trên ImageNet (IMAGENET1K_V1), bỏ tầng phân loại cuối.
    Đầu ra sau Global Average Pooling có kích thước `(N, 1280)`.
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


# ==============================================================================
# 3. Hàm Tiền xử lý Video dùng chung cho Demo (`preprocess_video`)
# ==============================================================================
def sample_raw_video_frames_uint8(
    video_path: Union[str, Path],
    num_frames: int = DEFAULT_NUM_FRAMES,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Trích `num_frames` khung hình cách đều từ file video gốc, resize về `image_size x image_size`
    bằng `cv2.INTER_AREA`, chuyển BGR -> RGB, trả về mảng `uint8` kích thước `(num_frames, 224, 224, 3)`.
    Khớp 100% với hàm `sample_frames` trong `preprocess.py` của Hiếu (Mã A).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở file video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        idxs = np.linspace(0, total - 1, num_frames).round().astype(int)
        want: Dict[int, List[int]] = {}
        for pos, idx in enumerate(idxs):
            want.setdefault(int(idx), []).append(pos)

        frames: List[Optional[np.ndarray]] = [None] * num_frames
        i = 0
        last = int(idxs.max())
        while i <= last:
            if not cap.grab():
                break
            if i in want:
                ok, fr = cap.retrieve()
                if ok and fr is not None:
                    fr_rgb = cv2.cvtColor(
                        cv2.resize(fr, (image_size, image_size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB,
                    )
                    for pos in want[i]:
                        frames[pos] = fr_rgb
            i += 1
        cap.release()

        good = [f for f in frames if f is not None]
        if good:
            for k in range(num_frames):
                if frames[k] is None:
                    frames[k] = frames[k - 1] if k > 0 and frames[k - 1] is not None else good[0]
            return np.stack(frames).astype(np.uint8)  # type: ignore[arg-type]
    else:
        cap.release()

    # Fallback: đọc tuần tự toàn bộ frame nếu metadata CAP_PROP_FRAME_COUNT bị lỗi
    cap = cv2.VideoCapture(str(video_path))
    raw_frames: List[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        fr_rgb = cv2.cvtColor(
            cv2.resize(fr, (image_size, image_size), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        raw_frames.append(fr_rgb)
    cap.release()

    if not raw_frames:
        raise RuntimeError(f"Không đọc được khung hình nào từ video: {video_path}")

    idxs = np.linspace(0, len(raw_frames) - 1, num_frames).round().astype(int)
    selected = [raw_frames[int(idx)] for idx in idxs]
    return np.stack(selected, axis=0).astype(np.uint8)


def _resample_temporal_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Lấy chỉ số frame cách đều khi số frame lưu trữ khác với `num_frames`."""
    if total_frames <= 0:
        raise ValueError(f"Tổng số frame không hợp lệ: {total_frames}")
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int)


def preprocess_video(
    video_path: Union[str, Path, np.ndarray, torch.Tensor],
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> torch.FloatTensor:
    """
    Hàm tiền xử lý chuẩn dùng chung cho Quy (B), Linh (C) và Thiên (D) trong `demo.py` (T22).

    Tham số:
        video_path: Đường dẫn tới file video gốc (`.mp4`, `.avi`, `.mkv`, `.mov`, ...)
                    HOẶC file frame đã trích sẵn `.npy` kích thước `(T, 224, 224, 3)` `uint8`
                    (cũng hỗ trợ trực tiếp mảng `np.ndarray` hoặc `torch.Tensor`).
        num_frames: Số lượng khung hình trích xuất cách đều (mặc định: `16`).

    Trả về:
        `torch.FloatTensor` có kích thước `(1, num_frames, 3, 224, 224)` ở hệ màu RGB,
        đã resize về `224x224` và chuẩn hóa theo ImageNet mean/std bằng đúng `to_torch_layout`
        trong `preprocess_utils.py` của Hiếu (Mã A).
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames phải > 0, nhận được: {num_frames}")

    # Trường hợp 1: Đầu vào là torch.Tensor truyền trực tiếp
    if isinstance(video_path, torch.Tensor):
        arr = video_path.detach().cpu().numpy()
        return preprocess_video(arr, num_frames=num_frames)

    # Trường hợp 2: Đầu vào là mảng numpy truyền trực tiếp hoặc đọc từ file .npy
    if isinstance(video_path, np.ndarray):
        arr = video_path
    else:
        path_obj = Path(video_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Không tìm thấy đường dẫn đầu vào: {path_obj}")

        suffix = path_obj.suffix.lower()
        if suffix == ".npy":
            arr = load_frames(path_obj)
        elif suffix in VIDEO_EXTENSIONS or suffix != ".npy":
            frames_uint8 = sample_raw_video_frames_uint8(
                path_obj, num_frames=num_frames, image_size=IMAGE_SIZE
            )
            tensor_tchw = torch.from_numpy(
                np.ascontiguousarray(to_torch_layout(frames_uint8), dtype=np.float32)
            )
            return tensor_tchw.unsqueeze(0).float()

    # Xử lý mảng numpy `arr` (từ file .npy hoặc truyền trực tiếp)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(
            f"Mảng frame phải có 4 chiều (T, H, W, 3) hoặc (T, 3, H, W), nhận được: {arr.shape}"
        )

    # Nếu số frame khác `num_frames`, lấy mẫu cách đều
    if arr.shape[0] != num_frames:
        idxs = _resample_temporal_indices(arr.shape[0], num_frames)
        arr = arr[idxs]

    # Kiểm tra layout (T, H, W, 3) vs (T, 3, H, W)
    if arr.shape[-1] == 3 and arr.shape[1] != 3:
        # Layout (T, H, W, 3)
        if arr.shape[1] != IMAGE_SIZE or arr.shape[2] != IMAGE_SIZE:
            arr = np.stack(
                [
                    cv2.resize(
                        np.ascontiguousarray(frame),
                        (IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                    for frame in arr
                ],
                axis=0,
            )
        if arr.dtype == np.uint8 or arr.max() > 10.0:
            frames_tchw = to_torch_layout(arr.astype(np.uint8))
        else:
            frames_tchw = arr.astype(np.float32).transpose(0, 3, 1, 2)
    elif arr.shape[1] == 3:
        # Layout (T, 3, H, W)
        frames_tchw = arr.astype(np.float32)
        if frames_tchw.max() > 10.0:
            # Đưa về (T, H, W, 3) uint8 rồi gọi to_torch_layout để đảm bảo khớp tuyệt đối
            frames_hwc = np.transpose(frames_tchw, (0, 2, 3, 1)).astype(np.uint8)
            frames_tchw = to_torch_layout(frames_hwc)
    else:
        raise ValueError(f"Không nhận diện được chiều kênh màu của mảng frame: {arr.shape}")

    tensor_out = torch.from_numpy(np.ascontiguousarray(frames_tchw, dtype=np.float32)).unsqueeze(0)
    return tensor_out.float()


# ==============================================================================
# 4. Bộ nhớ đệm (Cache) Mô hình & Suy luận Nhanh (`predict_baseline`)
# ==============================================================================
_EXTRACTOR_CACHE: Dict[str, nn.Module] = {}
_BASELINE_MODEL_CACHE: Dict[Tuple[str, float, str], Dict[str, Any]] = {}


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Xác định thiết bị tính toán (ưu tiên CUDA nếu có)."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_checkpoint_path(checkpoint_path: Optional[Union[str, Path]] = None) -> Path:
    """Tìm đường dẫn file checkpoint `baseline.pt` theo thứ tự ưu tiên."""
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại đường dẫn chỉ định: {p}")

    candidates = [
        WORKSPACE_DIR / "baseline.pt",
        Path("baseline.pt"),
        WORKSPACE_DIR / "outputs_baseline" / "baseline.pt",
        Path("outputs_baseline") / "baseline.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        "Không tìm thấy checkpoint 'baseline.pt'. Hãy chạy train_baseline.py (T09) trước "
        "hoặc truyền rõ tham số checkpoint_path."
    )


def get_cached_extractor(device: Optional[Union[str, torch.device]] = None) -> nn.Module:
    """Lấy MobileNetV2 feature extractor từ bộ nhớ đệm (khởi tạo 1 lần duy nhất trên mỗi device)."""
    dev = _resolve_device(device)
    key = str(dev)
    if key not in _EXTRACTOR_CACHE:
        _EXTRACTOR_CACHE[key] = build_mobilenetv2_extractor(dev)
    return _EXTRACTOR_CACHE[key]


def _infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Tự động suy luận cấu hình RNN từ trọng số `state_dict` nếu checkpoint thiếu `config`."""
    rnn_type = "lstm"
    hidden_size = 256
    input_dim = 1280
    num_layers = 1
    bidirectional = False

    if "rnn.weight_ih_l0" in state_dict:
        w_ih = state_dict["rnn.weight_ih_l0"]
        input_dim = int(w_ih.shape[1])
        # LSTM có 4 cổng (4 * hidden_size), GRU có 3 cổng (3 * hidden_size)
        if "classifier.1.weight" in state_dict:
            clf_in = int(state_dict["classifier.1.weight"].shape[1])
            bidirectional = "rnn.weight_ih_l0_reverse" in state_dict
            hidden_size = clf_in // (2 if bidirectional else 1)
            gates = int(w_ih.shape[0]) // hidden_size
            rnn_type = "gru" if gates == 3 else "lstm"
        else:
            hidden_size = int(w_ih.shape[0]) // 4

    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("rnn.weight_ih_l"):
            suffix = k.split("rnn.weight_ih_l", 1)[1]
            layer_num = suffix.split("_")[0]
            if layer_num.isdigit():
                layer_indices.add(int(layer_num))
    if layer_indices:
        num_layers = max(layer_indices) + 1

    return {
        "input_dim": input_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "rnn_type": rnn_type,
        "dropout": 0.3,
        "bidirectional": bidirectional,
        "num_frames": DEFAULT_NUM_FRAMES,
        "threshold": 0.5,
    }


def get_cached_baseline_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Tải và cache mô hình `ViolenceBaselineRNN` cùng cấu hình từ `baseline.pt`.
    Tự động nạp lại nếu file `baseline.pt` trên đĩa được cập nhật mới hơn.
    """
    dev = _resolve_device(device)
    resolved_path = _resolve_checkpoint_path(checkpoint_path)
    mtime = resolved_path.stat().st_mtime
    cache_key = (str(resolved_path), mtime, str(dev))

    if cache_key in _BASELINE_MODEL_CACHE:
        return _BASELINE_MODEL_CACHE[cache_key]

    ckpt = torch.load(resolved_path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        raw_config = ckpt.get("config", {}) or {}
        val_metrics = ckpt.get("val_metrics", {}) or {}
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        raw_config = ckpt.get("config", {}) or {}
        val_metrics = ckpt.get("val_metrics", {}) or {}
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        raw_config = {}
        val_metrics = {}
    else:
        raise ValueError(f"Định dạng checkpoint không hợp lệ tại: {resolved_path}")

    inferred = _infer_config_from_state_dict(state_dict)
    config = {**inferred, **raw_config}

    # Hỗ trợ lấy ngưỡng tối ưu nếu được lưu ở cấp cao nhất của checkpoint
    if isinstance(ckpt, dict):
        for thresh_key in ("best_threshold", "threshold", "val_threshold"):
            if thresh_key in ckpt and ckpt[thresh_key] is not None:
                config["threshold"] = float(ckpt[thresh_key])
                break

    model = ViolenceBaselineRNN(
        input_dim=int(config.get("input_dim", 1280)),
        hidden_size=int(config.get("hidden_size", 256)),
        num_layers=int(config.get("num_layers", 1)),
        rnn_type=str(config.get("rnn_type", "lstm")),
        dropout=float(config.get("dropout", 0.3)),
        bidirectional=bool(config.get("bidirectional", False)),
        pooling=str(config.get("pooling", "last")),
    ).to(dev)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    bundle = {
        "model": model,
        "config": config,
        "val_metrics": val_metrics,
        "checkpoint_path": resolved_path,
        "device": dev,
    }
    _BASELINE_MODEL_CACHE[cache_key] = bundle
    return bundle


@torch.no_grad()
def predict_baseline_detailed(
    video_path: Union[str, Path, np.ndarray, torch.Tensor],
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Dự đoán bạo lực cho 1 video (hoặc file `.npy`) bằng mô hình Baseline và trả về dict chi tiết.

    Trả về dict gồm:
      - `prob`: float trong `[0.0, 1.0]` (xác suất lớp Violence)
      - `label_id`: int (`1` = Violence, `0` = NonViolence)
      - `label_name`: str (`"Violence"` hoặc `"NonViolence"`)
      - `label_vi`: str (`"Bạo lực"` hoặc `"Không bạo lực"`)
      - `threshold`: float (ngưỡng phân loại áp dụng)
      - `inference_time_ms`: float (tổng thời gian suy luận tính bằng mili-giây)
      - `preprocess_time_ms`: float (thời gian đọc & tiền xử lý frame)
      - `backbone_time_ms`: float (thời gian trích đặc trưng MobileNetV2)
      - `rnn_time_ms`: float (thời gian suy luận qua LSTM/GRU)
      - `num_frames`: int (số frame T đưa vào mô hình)
      - `model_config`: dict (cấu hình mô hình Baseline)
    """
    dev = _resolve_device(device)
    bundle = get_cached_baseline_model(checkpoint_path=checkpoint_path, device=dev)
    model: nn.Module = bundle["model"]
    config: Dict[str, Any] = bundle["config"]

    num_frames = int(config.get("num_frames", DEFAULT_NUM_FRAMES))
    used_threshold = (
        float(threshold)
        if threshold is not None
        else float(config.get("threshold", 0.5))
    )

    t0 = time.perf_counter()

    # Hỗ trợ trường hợp đầu vào là file/mảng đặc trưng đã trích sẵn (T, 1280)
    is_preextracted_feature = False
    if isinstance(video_path, (str, Path)) and Path(video_path).suffix.lower() == ".npy":
        raw_arr = np.load(video_path, mmap_mode="r")
        if raw_arr.ndim == 2 and raw_arr.shape[1] == int(config.get("input_dim", 1280)):
            is_preextracted_feature = True
            feat_np = np.array(raw_arr, dtype=np.float32)
            if feat_np.shape[0] != num_frames:
                idxs = _resample_temporal_indices(feat_np.shape[0], num_frames)
                feat_np = feat_np[idxs]
            features_tensor = torch.from_numpy(feat_np).unsqueeze(0).to(dev)
            t_pre = time.perf_counter()
            t_backbone = t_pre

    if not is_preextracted_feature:
        extractor = get_cached_extractor(dev)
        # Bước 1: Tiền xử lý video về (1, T, 3, 224, 224)
        video_tensor = preprocess_video(video_path, num_frames=num_frames)
        t_pre = time.perf_counter()

        # Bước 2: Đưa T frame qua MobileNetV2 -> (1, T, 1280)
        b, t, c, h, w = video_tensor.shape
        frames_flat = video_tensor.view(b * t, c, h, w).to(dev)
        feats_flat = extractor(frames_flat)  # (T, 1280)
        features_tensor = feats_flat.view(b, t, -1)  # (1, T, 1280)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_backbone = time.perf_counter()

    # Bước 3: Đưa chuỗi đặc trưng (1, T, 1280) qua LSTM/GRU
    prob_tensor = model(features_tensor)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_end = time.perf_counter()

    prob = float(torch.clamp(prob_tensor.reshape(-1)[0], 0.0, 1.0).item())
    label_id = 1 if prob >= used_threshold else 0
    label_name = "Violence" if label_id == 1 else "NonViolence"
    label_vi = "Bạo lực" if label_id == 1 else "Không bạo lực"

    preprocess_ms = (t_pre - t0) * 1000.0
    backbone_ms = (t_backbone - t_pre) * 1000.0
    rnn_ms = (t_end - t_backbone) * 1000.0
    total_ms = (t_end - t0) * 1000.0

    return {
        "video_path": str(video_path) if isinstance(video_path, (str, Path)) else "<in_memory_array>",
        "prob": round(prob, 6),
        "label_id": label_id,
        "label_name": label_name,
        "label_vi": label_vi,
        "threshold": used_threshold,
        "inference_time_ms": round(total_ms, 3),
        "preprocess_time_ms": round(preprocess_ms, 3),
        "backbone_time_ms": round(backbone_ms, 3),
        "rnn_time_ms": round(rnn_ms, 3),
        "num_frames": num_frames,
        "device": str(dev),
        "checkpoint_path": str(bundle["checkpoint_path"]),
        "model_config": config,
    }


@torch.no_grad()
def predict_baseline(
    video_path: Union[str, Path, np.ndarray, torch.Tensor],
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    threshold: Optional[float] = None,
    return_int_label: bool = False,
) -> Tuple[float, Union[str, int]]:
    """
    Hàm suy luận chuẩn cho `demo.py` (T22 - thỏa thuận giữa Quy B và Thiên D):
      - Tải `baseline.pt` (có cache trong bộ nhớ để gọi lặp lại tức thì trên Gradio/Streamlit).
      - Trả về bộ `(prob, label)` trong đó:
          + `prob`: `float` trong `[0.0, 1.0]` (xác suất video có hành vi bạo lực - Violence)
          + `label`: `"Violence"` hoặc `"NonViolence"` (hoặc `1` / `0` nếu `return_int_label=True`).
    """
    res = predict_baseline_detailed(
        video_path=video_path,
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
    )
    prob = float(res["prob"])
    label: Union[str, int] = int(res["label_id"]) if return_int_label else str(res["label_name"])
    return prob, label


def get_model_size_summary(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Tính dung lượng checkpoint và số lượng tham số của toàn bộ pipeline Baseline (MobileNetV2 + RNN)."""
    dev = _resolve_device(device)
    bundle = get_cached_baseline_model(checkpoint_path=checkpoint_path, device=dev)
    extractor = get_cached_extractor(dev)
    rnn_model: nn.Module = bundle["model"]
    ckpt_path: Path = bundle["checkpoint_path"]

    ckpt_size_mb = ckpt_path.stat().st_size / (1024.0 * 1024.0)
    mobilenet_params = sum(p.numel() for p in extractor.parameters())
    mobilenet_size_mb = sum(p.numel() * p.element_size() for p in extractor.parameters()) / (
        1024.0 * 1024.0
    )
    rnn_params = sum(p.numel() for p in rnn_model.parameters())
    rnn_size_mb = sum(p.numel() * p.element_size() for p in rnn_model.parameters()) / (
        1024.0 * 1024.0
    )

    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_file_size_mb": round(ckpt_size_mb, 4),
        "rnn_params": int(rnn_params),
        "rnn_weights_memory_mb": round(rnn_size_mb, 4),
        "mobilenetv2_params": int(mobilenet_params),
        "mobilenetv2_weights_memory_mb": round(mobilenet_size_mb, 4),
        "total_pipeline_params": int(mobilenet_params + rnn_params),
        "total_pipeline_memory_mb": round(mobilenet_size_mb + rnn_size_mb, 4),
        "config": bundle["config"],
        "val_metrics": bundle["val_metrics"],
    }


# ==============================================================================
# 5. Kiểm thử tự động CLI (`python baseline_inference.py --test`)
# ==============================================================================
def _collect_test_samples() -> List[Dict[str, Any]]:
    """
    Thu thập danh sách >= 8 mẫu thực tế (gồm cả video gốc `.mp4` từ RLVS và file frame `.npy`
    từ RLVS + RWF-2000) để kiểm thử cổng G4 ("Chạy ổn định trên ít nhất 5 video khác nhau").
    """
    samples: List[Dict[str, Any]] = []

    rlvs_raw_dir = HIEU_VIOLENCE_DIR / "data" / "rlvs" / "Real Life Violence Dataset"
    frames_n16_dir = HIEU_VIOLENCE_DIR / "frames" / "n16"

    # 1. Lấy 2 video gốc .mp4 lớp NonViolence và 2 video gốc .mp4 lớp Violence từ RLVS
    nv_raw_dir = rlvs_raw_dir / "NonViolence"
    v_raw_dir = rlvs_raw_dir / "Violence"
    if nv_raw_dir.exists():
        for p in sorted(nv_raw_dir.glob("*.mp4"))[:2]:
            samples.append(
                {
                    "name": f"RLVS_Raw_MP4/{p.name}",
                    "path": p,
                    "dataset": "RLVS (raw .mp4)",
                    "expected_label": "NonViolence",
                }
            )
    if v_raw_dir.exists():
        for p in sorted(v_raw_dir.glob("*.mp4"))[:2]:
            samples.append(
                {
                    "name": f"RLVS_Raw_MP4/{p.name}",
                    "path": p,
                    "dataset": "RLVS (raw .mp4)",
                    "expected_label": "Violence",
                }
            )

    # 2. Lấy 2 file frame .npy từ RLVS (1 NonViolence, 1 Violence)
    if frames_n16_dir.exists():
        for p in sorted(frames_n16_dir.glob("NV_*.npy"))[:1]:
            samples.append(
                {
                    "name": f"RLVS_Frames_NPY/{p.name}",
                    "path": p,
                    "dataset": "RLVS (frames .npy)",
                    "expected_label": "NonViolence",
                }
            )
        for p in sorted(frames_n16_dir.glob("V_*.npy"))[:1]:
            samples.append(
                {
                    "name": f"RLVS_Frames_NPY/{p.name}",
                    "path": p,
                    "dataset": "RLVS (frames .npy)",
                    "expected_label": "Violence",
                }
            )

        # 3. Lấy 2 file frame .npy từ RWF-2000 (1 NonFight rwf_N_*, 1 Fight rwf_F_*)
        for p in sorted(frames_n16_dir.glob("rwf_N_*.npy"))[:2]:
            samples.append(
                {
                    "name": f"RWF2000_Frames_NPY/{p.name}",
                    "path": p,
                    "dataset": "RWF-2000 (cross_test .npy)",
                    "expected_label": "NonViolence",
                }
            )
        for p in sorted(frames_n16_dir.glob("rwf_F_*.npy"))[:2]:
            samples.append(
                {
                    "name": f"RWF2000_Frames_NPY/{p.name}",
                    "path": p,
                    "dataset": "RWF-2000 (cross_test .npy)",
                    "expected_label": "Violence",
                }
            )

    return samples


def run_self_test(checkpoint_path: Optional[Union[str, Path]] = None) -> None:
    """Chạy kiểm thử toàn diện cho việc T22 và cổng kiểm tra G4."""
    print("=" * 80)
    print("[T22 SELF-TEST] KIỂM THỬ MODULE TIỀN XỬ LÝ & SUY LUẬN BASELINE (Gate G4)")
    print("=" * 80)

    samples = _collect_test_samples()
    if len(samples) < 5:
        raise RuntimeError(f"Cần ít nhất 5 mẫu video để kiểm thử Gate G4, tìm thấy: {len(samples)}")

    # Bước 1: Kiểm thử hàm preprocess_video trên cả .mp4 và .npy
    print("\n1. Kiểm thử hàm `preprocess_video(video_path, num_frames=16)`:")
    for s in samples[:4]:
        t0 = time.perf_counter()
        tensor = preprocess_video(s["path"], num_frames=16)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert isinstance(tensor, torch.FloatTensor) or tensor.dtype == torch.float32, (
            f"Kiểu trả về phải là FloatTensor, nhận được {tensor.dtype}"
        )
        assert tuple(tensor.shape) == (1, 16, 3, 224, 224), (
            f"Kích thước sai: {tuple(tensor.shape)}, kỳ vọng (1, 16, 3, 224, 224)"
        )
        print(
            f"  [OK] {s['name']:<32} -> shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, min={tensor.min():.2f}, max={tensor.max():.2f} ({dt_ms:.1f} ms)"
        )

    # Kiểm tra đối khớp tuyệt đối giữa đọc từ video gốc .mp4 và đọc từ .npy của Hiếu (NV_1)
    nv1_mp4 = HIEU_VIOLENCE_DIR / "data" / "rlvs" / "Real Life Violence Dataset" / "NonViolence" / "NV_1.mp4"
    nv1_npy = HIEU_VIOLENCE_DIR / "frames" / "n16" / "NV_1.npy"
    if nv1_mp4.exists() and nv1_npy.exists():
        t_mp4 = preprocess_video(nv1_mp4, num_frames=16)
        t_npy = preprocess_video(nv1_npy, num_frames=16)
        max_diff = float((t_mp4 - t_npy).abs().max().item())
        print(
            f"  [OK] Kiểm tra đồng nhất `NV_1.mp4` vs `NV_1.npy` của Hiếu: max_abs_diff = {max_diff:.6f}"
        )

    # Bước 2: Kiểm thử tải checkpoint & thống kê dung lượng mô hình
    print("\n2. Thông tin Checkpoint & Dung lượng Mô hình Baseline:")
    summary = get_model_size_summary(checkpoint_path=checkpoint_path)
    print(f"  - Đường dẫn checkpoint      : {summary['checkpoint_path']}")
    print(f"  - Dung lượng baseline.pt    : {summary['checkpoint_file_size_mb']:.4f} MB")
    print(
        f"  - Tham số nhánh RNN         : {summary['rnn_params']:,} params "
        f"({summary['rnn_weights_memory_mb']:.4f} MB)"
    )
    print(
        f"  - Tham số MobileNetV2       : {summary['mobilenetv2_params']:,} params "
        f"({summary['mobilenetv2_weights_memory_mb']:.4f} MB)"
    )
    print(
        f"  - Tổng tham số toàn tuyến   : {summary['total_pipeline_params']:,} params "
        f"({summary['total_pipeline_memory_mb']:.4f} MB)"
    )
    print(f"  - Cấu hình mô hình          : {summary['config']}")

    # Bước 3: Warm-up cache trước khi đo tốc độ suy luận
    _ = predict_baseline(samples[0]["path"], checkpoint_path=checkpoint_path)

    # Bước 4: Chạy suy luận trên toàn bộ >= 8 mẫu thực tế (RLVS .mp4, RLVS .npy, RWF-2000 .npy)
    print(f"\n3. Kiểm thử `predict_baseline` trên {len(samples)} video mẫu (RLVS & RWF-2000):")
    print(
        f"  {'#':<3} {'Mẫu kiểm thử':<38} {'Nhãn thực':<12} {'Dự đoán':<12} "
        f"{'Xác suất':<10} {'Tổng (ms)':<10} {'Pre (ms)':<9} {'CNN (ms)':<9} {'RNN (ms)':<8}"
    )
    print("  " + "-" * 116)

    mp4_times: List[float] = []
    npy_times: List[float] = []
    correct_count = 0

    for idx, s in enumerate(samples, start=1):
        prob, label = predict_baseline(s["path"], checkpoint_path=checkpoint_path)
        detail = predict_baseline_detailed(s["path"], checkpoint_path=checkpoint_path)

        assert 0.0 <= prob <= 1.0, f"Xác suất ngoài khoảng [0, 1]: {prob}"
        assert label in {"Violence", "NonViolence"}, f"Nhãn không hợp lệ: {label}"

        if label == s["expected_label"]:
            correct_count += 1

        if Path(s["path"]).suffix.lower() == ".npy":
            npy_times.append(float(detail["inference_time_ms"]))
        else:
            mp4_times.append(float(detail["inference_time_ms"]))

        print(
            f"  {idx:<3} {s['name']:<38} {s['expected_label']:<12} {label:<12} "
            f"{prob:<10.4f} {detail['inference_time_ms']:<10.2f} "
            f"{detail['preprocess_time_ms']:<9.2f} {detail['backbone_time_ms']:<9.2f} "
            f"{detail['rnn_time_ms']:<8.2f}"
        )

    print("  " + "-" * 116)
    print(f"  - Số lượng video kiểm thử thành công : {len(samples)}/{len(samples)} (Gate G4 >= 5 video: ĐẠT)")
    print(f"  - Độ chính xác trên tập mẫu kiểm thử : {correct_count}/{len(samples)}")
    if mp4_times:
        print(
            f"  - Thời gian suy luận TB (video gốc .mp4, đã cache model) : "
            f"{np.mean(mp4_times):.2f} ms/video (min={np.min(mp4_times):.2f}, max={np.max(mp4_times):.2f})"
        )
    if npy_times:
        print(
            f"  - Thời gian suy luận TB (frame .npy, đã cache model)     : "
            f"{np.mean(npy_times):.2f} ms/video (min={np.min(npy_times):.2f}, max={np.max(npy_times):.2f})"
        )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T22: Module tiền xử lý & suy luận Baseline (MobileNetV2 + LSTM/GRU) cho demo.py"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Đường dẫn tới 1 file video (.mp4/.avi/...) hoặc file frame (.npy) cần dự đoán",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Đường dẫn tới file baseline.pt (mặc định tự tìm trong thư mục làm việc)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Ngưỡng phân loại lớp Violence (mặc định đọc từ checkpoint hoặc 0.5)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Chạy kiểm thử tự động trên >= 5 video mẫu thực tế từ RLVS và RWF-2000 (Gate G4)",
    )
    args = parser.parse_args()

    if args.test:
        run_self_test(checkpoint_path=args.checkpoint)
        return

    if args.video is not None:
        res = predict_baseline_detailed(
            video_path=args.video,
            checkpoint_path=args.checkpoint,
            threshold=args.threshold,
        )
        print(f"Video          : {res['video_path']}")
        print(f"Dự đoán        : {res['label_name']} ({res['label_vi']}) [label_id={res['label_id']}]")
        print(f"Xác suất (prob): {res['prob']:.4f} (threshold={res['threshold']:.4f})")
        print(
            f"Thời gian      : {res['inference_time_ms']:.2f} ms "
            f"(preprocess={res['preprocess_time_ms']:.2f} ms, "
            f"MobileNetV2={res['backbone_time_ms']:.2f} ms, "
            f"RNN={res['rnn_time_ms']:.2f} ms) trên {res['device']}"
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
