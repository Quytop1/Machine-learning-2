"""
T09, T10, T14, T19, T20: Huấn luyện mô hình Baseline (MobileNetV2 + LSTM/GRU),
Thí nghiệm biến thể (Ablation Study), Đánh giá Tăng cường dữ liệu (Data Augmentation),
Xuất dự đoán chuẩn cho Thiên (Mã D) và Đánh giá chéo ngoài phân phối trên RWF-2000.
Phụ trách: Nguyễn Xuân Quy (Mã B)

Đầu ra:
  1. Checkpoint tốt nhất: baseline.pt (và outputs_baseline/baseline.pt)
  2. Thư mục dự đoán preds_baseline/:
     - preds_baseline_val.csv (RLVS val)
     - preds_baseline_test.csv (RLVS test)
     - preds_baseline.csv (gộp val + test, đồng thời lưu bản sao tại gốc preds_baseline.csv)
     - preds_baseline_cross_test.csv (RWF-2000 cross_test cho T20)
  3. Bảng thí nghiệm biến thể T10: ablation_baseline.csv (và outputs_baseline/ablation_baseline.csv)
  4. Bảng đối chiếu tăng cường dữ liệu T14: outputs_baseline/t14_augmentation_baseline.csv
  5. Các biểu đồ trực quan hóa nhãn tiếng Việt:
     - loss_curve.png
     - confusion_matrix_baseline.png
     - roc_curve_baseline.png
     - ablation_comparison.png
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    """Cố định random seed để kết quả huấn luyện và so sánh công bằng, tái lập được."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InMemoryFeatureDataset(Dataset):
    """Dataset nạp đặc trưng (N, T, 1280) trong bộ nhớ giúp huấn luyện cực nhanh."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        video_ids: List[str],
        splits: List[str],
        num_frames: int = 16,
    ) -> None:
        assert len(features) == len(labels) == len(video_ids) == len(splits)
        if features.shape[1] != num_frames:
            indices = np.linspace(0, features.shape[1] - 1, num_frames, dtype=int)
            features = features[:, indices, :]
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.video_ids = [str(v) for v in video_ids]
        self.splits = [str(s) for s in splits]

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        return (
            self.features[idx],
            self.labels[idx],
            self.video_ids[idx],
            self.splits[idx],
        )


class VideoFeatureDataset(Dataset):
    """Dataset đọc trực tiếp các file đặc trưng .npy kích thước (T, 1280) đã lưu ở T08."""

    def __init__(self, df: pd.DataFrame, num_frames: int = 16) -> None:
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.df.iloc[idx]
        feat_path = Path(str(row["feature_path"]))
        feat = np.load(feat_path).astype(np.float32)  # (T_saved, 1280)

        if feat.shape[0] != self.num_frames:
            indices = np.linspace(0, feat.shape[0] - 1, self.num_frames, dtype=int)
            feat = feat[indices]

        label = float(row["label"])
        video_id = str(row["video_id"])
        split = str(row["split"])
        return (
            torch.from_numpy(feat),
            torch.tensor(label, dtype=torch.float32),
            video_id,
            split,
        )


class ViolenceBaselineRNN(nn.Module):
    """
    Kiến trúc Baseline theo mục 6.2 (T09):
      Input (B, T, 1280) -> LSTM hoặc GRU (1-2 lớp) -> Dense + Sigmoid -> Xác suất lớp Violence [0, 1]
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_size: int = 256,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"rnn_type chỉ nhận 'lstm' hoặc 'gru', nhận được: {rnn_type}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, T, 1280)
        rnn_out, _ = self.rnn(x)  # (batch_size, T, hidden_size)
        last_hidden = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        probs = self.classifier(last_hidden).squeeze(-1)  # (batch_size,)
        return probs


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Tính bộ chỉ số theo yêu cầu mục 3 & mục 6.4:
    Ưu tiên Recall của lớp Violence và F1, bên cạnh Accuracy, Precision và ROC-AUC.
    """
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    except ValueError:
        auc = 0.0
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_violence": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


def find_optimal_threshold_on_val(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Tìm ngưỡng tối ưu chỉ trên tập val (tối đa hóa F1 + ưu tiên Recall Violence).
    Tuân thủ nghiêm ngặt quy tắc của Hiếu: chỉ chọn ngưỡng trên val RLVS.
    """
    best_th = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    best_score = best_metrics["f1"]

    for th in np.linspace(0.30, 0.70, 41):
        m = compute_metrics(y_true, y_prob, threshold=float(th))
        if m["f1"] > best_score + 1e-5 or (
            abs(m["f1"] - best_score) <= 1e-5
            and m["recall_violence"] > best_metrics["recall_violence"]
        ):
            best_score = m["f1"]
            best_th = float(round(th, 2))
            best_metrics = m
    return best_th, best_metrics


def load_feature_split_data(
    features_dir: Path,
    num_frames: int = 16,
    is_cross_test: bool = False,
    is_augmented_train: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], List[str]]]:
    """
    Hỗ trợ nạp đặc trưng linh hoạt từ:
      1. File .npz (vd: features_n16.npz, features_T16.npz, features_cross_n16.npz, features_train_aug_n16.npz)
      2. File .csv chỉ mục + các file .npy từng video (vd: features_n16.csv, features_T16.csv)
    Trả về (features, labels, video_ids, splits) hoặc None nếu không tìm thấy.
    """
    if is_augmented_train:
        npz_candidates = [
            features_dir / f"features_train_aug_n{num_frames}.npz",
            features_dir / f"features_aug_n{num_frames}.npz",
            features_dir / f"features_train_aug_T{num_frames}.npz",
            features_dir / f"features_n{num_frames}_aug.npz",
            features_dir / "features_train_aug.npz",
        ]
        csv_candidates = [
            features_dir / f"features_train_aug_n{num_frames}.csv",
            features_dir / f"features_aug_n{num_frames}.csv",
            features_dir / f"features_train_aug_T{num_frames}.csv",
            features_dir / f"features_T{num_frames}_aug.csv",
        ]
    elif is_cross_test:
        npz_candidates = [
            features_dir / f"features_cross_n{num_frames}.npz",
            features_dir / f"features_cross_T{num_frames}.npz",
            features_dir / f"cross_test_n{num_frames}.npz",
            features_dir / f"rwf_n{num_frames}.npz",
        ]
        csv_candidates = [
            features_dir / f"features_cross_n{num_frames}.csv",
            features_dir / f"features_cross_T{num_frames}.csv",
            features_dir / f"cross_test_n{num_frames}.csv",
        ]
    else:
        npz_candidates = [
            features_dir / f"features_n{num_frames}.npz",
            features_dir / f"features_T{num_frames}.npz",
            features_dir / f"rlvs_n{num_frames}.npz",
        ]
        csv_candidates = [
            features_dir / f"features_n{num_frames}.csv",
            features_dir / f"features_T{num_frames}.csv",
        ]

    # 1. Thử nạp từ .npz trước
    if is_augmented_train:
        main_npz = features_dir / f"features_n{num_frames}.npz"
        if main_npz.exists():
            data = np.load(main_npz, allow_pickle=True)
            if "train_aug_features" in data:
                feats = data["train_aug_features"].astype(np.float32)
                labels = data["train_aug_labels"].astype(np.float32)
                video_ids = [str(x) for x in data["train_aug_video_ids"].tolist()]
                splits = ["train"] * len(video_ids)
                return feats, labels, video_ids, splits

    for npz_path in npz_candidates:
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            feats = data["features"].astype(np.float32)
            labels = data["labels"].astype(np.float32)
            video_ids = [str(x) for x in data["video_ids"].tolist()]
            if "splits" in data:
                splits = [str(x) for x in data["splits"].tolist()]
            else:
                default_split = (
                    "train"
                    if is_augmented_train
                    else ("cross_test" if is_cross_test else "train")
                )
                splits = [default_split] * len(video_ids)
            return feats, labels, video_ids, splits

    # 2. Thử nạp từ .csv
    for csv_path in csv_candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if is_augmented_train and "is_aug" in df.columns:
                df = df[df["is_aug"] == 1].reset_index(drop=True)
            if is_cross_test and "split" in df.columns and any(
                s in {"cross", "cross_test", "rwf", "rwf_test"} for s in df["split"].astype(str).values
            ):
                df = df[df["split"].astype(str).isin({"cross", "cross_test", "rwf", "rwf_test"})].reset_index(drop=True)
            feats_list = []
            for p in df["feature_path"]:
                arr = np.load(Path(str(p))).astype(np.float32)
                if arr.shape[0] != num_frames:
                    idx = np.linspace(0, arr.shape[0] - 1, num_frames, dtype=int)
                    arr = arr[idx]
                feats_list.append(arr)
            feats = np.stack(feats_list, axis=0)
            labels = df["label"].to_numpy(dtype=np.float32)
            video_ids = [str(x) for x in df["video_id"].tolist()]
            if "split" in df.columns:
                splits = [str(x) for x in df["split"].tolist()]
            else:
                default_split = "cross_test" if is_cross_test else "train"
                splits = [default_split] * len(video_ids)
            return feats, labels, video_ids, splits

    # 3. Nếu đang tìm cross_test mà file chính (features_n{T}.npz hoặc .csv) có chứa sẵn split == 'cross' / 'cross_test'
    if is_cross_test:
        main_data = load_feature_split_data(
            features_dir, num_frames=num_frames, is_cross_test=False, is_augmented_train=False
        )
        if main_data is not None:
            feats, labels, video_ids, splits = main_data
            mask = [s in {"cross", "cross_test", "rwf", "rwf_test"} for s in splits]
            if any(mask):
                idx = np.where(mask)[0]
                return (
                    feats[idx],
                    labels[idx],
                    [video_ids[i] for i in idx],
                    ["cross_test" for _ in idx],
                )

    return None


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """Đánh giá mô hình trên một DataLoader và trả về (loss, metrics, bảng dự đoán chi tiết)."""
    model.eval()
    total_loss = 0.0
    total_count = 0

    all_ids: List[str] = []
    all_splits: List[str] = []
    all_true: List[int] = []
    all_prob: List[float] = []

    for x, y, video_ids, splits in loader:
        x = x.to(device)
        y = y.to(device)
        probs = model(x)
        probs_clamped = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        loss = criterion(probs_clamped, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        all_ids.extend(video_ids)
        all_splits.extend(splits)
        all_true.extend(y.cpu().numpy().astype(int).tolist())
        all_prob.extend(probs.cpu().numpy().astype(float).tolist())

    avg_loss = total_loss / max(total_count, 1)
    y_true_np = np.array(all_true, dtype=int)
    y_prob_np = np.array(all_prob, dtype=float)
    y_pred_np = (y_prob_np >= threshold).astype(int)

    metrics = compute_metrics(y_true_np, y_prob_np, threshold=threshold)
    preds_df = pd.DataFrame(
        {
            "video_id": all_ids,
            "split": all_splits,
            "y_true": y_true_np,
            "y_prob": np.round(y_prob_np, 6),
            "y_pred": y_pred_np,
        }
    )
    return avg_loss, metrics, preds_df


def plot_training_curves(history: Dict[str, List[float]], save_paths: List[Path]) -> None:
    """Vẽ và lưu biểu đồ đường cong Train/Val Loss và Val F1/Recall/Accuracy bằng tiếng Việt."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    # Biểu đồ 1: Đường cong BCE Loss
    axes[0].plot(
        epochs, history["train_loss"], label="Train Loss (BCE)", marker="o", markersize=4, linewidth=2
    )
    axes[0].plot(
        epochs, history["val_loss"], label="Validation Loss (BCE)", marker="s", markersize=4, linewidth=2
    )
    if "best_epoch" in history and len(history["best_epoch"]) > 0:
        best_ep = int(history["best_epoch"][0])
        axes[0].axvline(
            best_ep, color="red", linestyle=":", linewidth=1.5, label=f"Epoch tốt nhất ({best_ep})"
        )
    axes[0].set_title("Đường cong hàm mất mát (BCE Loss) theo Epoch", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Giá trị hàm mất mát (BCE Loss)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best")

    # Biểu đồ 2: Các chỉ số trên tập Validation
    axes[1].plot(
        epochs, history["val_f1"], label="Val F1-Score", marker="o", markersize=4, linewidth=2
    )
    axes[1].plot(
        epochs,
        history["val_recall"],
        label="Val Recall (Violence)",
        marker="^",
        markersize=4,
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        history["val_acc"],
        label="Val Accuracy",
        linestyle="--",
        linewidth=1.8,
    )
    if "train_acc" in history and len(history["train_acc"]) == len(epochs):
        axes[1].plot(
            epochs,
            history["train_acc"],
            label="Train Accuracy",
            linestyle=":",
            alpha=0.75,
            linewidth=1.5,
        )
    axes[1].set_title("Hiệu năng mô hình Baseline theo Epoch", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Điểm số (0.0 - 1.0)", fontsize=11)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    preds_dict: Dict[str, pd.DataFrame],
    save_paths: List[Path],
) -> None:
    """Vẽ ma trận nhầm lẫn trên tập Val, Test RLVS và Cross-test RWF-2000 với nhãn tiếng Việt."""
    n_plots = len(preds_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.2 * n_plots, 4.6), dpi=150)
    if n_plots == 1:
        axes = [axes]

    class_names = ["Không bạo lực\n(NonViolence=0)", "Bạo lực\n(Violence=1)"]

    for ax, (title_vi, df_sub) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(df_sub["y_true"], df_sub["y_pred"], labels=[0, 1])
        cm_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_pct = cm / cm_sum * 100.0

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title_vi, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=10)
        ax.set_ylabel("Nhãn thực tế (Actual)", fontsize=10)

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                text_val = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"
                ax.text(
                    j,
                    i,
                    text_val,
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.suptitle(
        "Ma trận nhầm lẫn của mô hình Baseline (MobileNetV2 + RNN)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    preds_dict: Dict[str, pd.DataFrame],
    save_paths: List[Path],
) -> None:
    """Vẽ đường cong ROC trên tập Val, Test RLVS và Cross-test RWF-2000 bằng tiếng Việt."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    styles = [
        {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.2},
        {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.2},
        {"color": "#d62728", "linestyle": "-.", "linewidth": 2.2},
    ]

    for idx, (label_vi, df_sub) in enumerate(preds_dict.items()):
        if len(np.unique(df_sub["y_true"])) < 2:
            continue
        fpr, tpr, _ = roc_curve(df_sub["y_true"], df_sub["y_prob"])
        auc_val = roc_auc_score(df_sub["y_true"], df_sub["y_prob"])
        st = styles[idx % len(styles)]
        ax.plot(
            fpr,
            tpr,
            label=f"{label_vi} (ROC-AUC = {auc_val:.4f})",
            **st,
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Ngẫu nhiên (AUC = 0.5000)")
    ax.set_title(
        "Đường cong ROC của mô hình Baseline trên các tập đánh giá",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Tỷ lệ Dương tính giả - FPR (False Positive Rate)", fontsize=11)
    ax.set_ylabel("Tỷ lệ Dương tính thật - TPR / Recall (Violence)", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_comparison(
    ablation_df: pd.DataFrame,
    save_paths: List[Path],
) -> None:
    """Vẽ biểu đồ so sánh F1, Recall (Violence) và ROC-AUC giữa các cấu hình Ablation (T10)."""
    labels = [
        f"{row['Mô hình (LSTM/GRU)']}\nT={row['Số frame (T)']}, H={row['Hidden size']}"
        for _, row in ablation_df.iterrows()
    ]
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 1.35), 5.5), dpi=150)
    rects1 = ax.bar(
        x - width,
        ablation_df["F1"],
        width,
        label="F1-Score (Val)",
        color="#1f77b4",
    )
    rects2 = ax.bar(
        x,
        ablation_df["Recall (Violence)"],
        width,
        label="Recall Violence (Val)",
        color="#ff7f0e",
    )
    rects3 = ax.bar(
        x + width,
        ablation_df["ROC-AUC"],
        width,
        label="ROC-AUC (Val)",
        color="#2ca02c",
    )

    ax.set_title(
        "So sánh hiệu năng các cấu hình Biến thể (Ablation Study - T10) trên tập Validation",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Cấu hình mô hình (Loại mạng tuần tự, Số frame T, Kích thước ẩn H)", fontsize=11)
    ax.set_ylabel("Điểm số trên tập Validation (0.0 - 1.0)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)

    min_val = max(0.0, float(ablation_df[["F1", "Recall (Violence)", "ROC-AUC"]].min().min()) - 0.12)
    ax.set_ylim(min_val, 1.04)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90 if len(labels) > 7 else 0,
            )

    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def build_dataloaders_for_config(
    features_dir: Path,
    fallback_df: Optional[pd.DataFrame],
    num_frames: int = 16,
    batch_size: int = 32,
    use_augmented_train: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Khởi tạo (train_loader, val_loader, test_loader, cross_test_loader) cho cấu hình num_frames.
    Ưu tiên nạp đặc trưng trích đúng ở T=num_frames (features_n8, features_n16, features_n20).
    Nếu bật use_augmented_train=True (cho T14), gộp thêm bản đặc trưng tăng cường của tập train.
    """
    split_data = load_feature_split_data(
        features_dir=features_dir,
        num_frames=num_frames,
        is_cross_test=False,
        is_augmented_train=False,
    )

    if split_data is not None:
        feats, labels, video_ids, splits = split_data
        splits_arr = np.array(splits)

        train_mask = splits_arr == "train"
        val_mask = splits_arr == "val"
        test_mask = splits_arr == "test"

        train_feats = feats[train_mask]
        train_labels = labels[train_mask]
        train_ids = [video_ids[i] for i in np.where(train_mask)[0]]
        train_splits = ["train"] * len(train_ids)

        if use_augmented_train:
            aug_data = load_feature_split_data(
                features_dir=features_dir,
                num_frames=num_frames,
                is_cross_test=False,
                is_augmented_train=True,
            )
            if aug_data is not None:
                aug_f, aug_l, aug_ids, _ = aug_data
                train_feats = np.concatenate([train_feats, aug_f], axis=0)
                train_labels = np.concatenate([train_labels, aug_l], axis=0)
                train_ids = train_ids + [f"{vid}_aug" for vid in aug_ids]
                train_splits = ["train"] * len(train_ids)

        train_ds = InMemoryFeatureDataset(
            train_feats, train_labels, train_ids, train_splits, num_frames=num_frames
        )
        val_ds = InMemoryFeatureDataset(
            feats[val_mask],
            labels[val_mask],
            [video_ids[i] for i in np.where(val_mask)[0]],
            ["val"] * int(val_mask.sum()),
            num_frames=num_frames,
        )
        test_ds = (
            InMemoryFeatureDataset(
                feats[test_mask],
                labels[test_mask],
                [video_ids[i] for i in np.where(test_mask)[0]],
                ["test"] * int(test_mask.sum()),
                num_frames=num_frames,
            )
            if test_mask.any()
            else None
        )
    elif fallback_df is not None:
        train_df = fallback_df[fallback_df["split"] == "train"].copy()
        val_df = fallback_df[fallback_df["split"] == "val"].copy()
        test_df = fallback_df[fallback_df["split"] == "test"].copy()

        train_ds = VideoFeatureDataset(train_df, num_frames=num_frames)
        val_ds = VideoFeatureDataset(val_df, num_frames=num_frames)
        test_ds = VideoFeatureDataset(test_df, num_frames=num_frames) if not test_df.empty else None
    else:
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đặc trưng tại {features_dir} cho T={num_frames}."
        )

    cross_data = load_feature_split_data(
        features_dir=features_dir,
        num_frames=num_frames,
        is_cross_test=True,
        is_augmented_train=False,
    )
    cross_ds = (
        InMemoryFeatureDataset(
            cross_data[0],
            cross_data[1],
            cross_data[2],
            ["cross_test"] * len(cross_data[2]),
            num_frames=num_frames,
        )
        if cross_data is not None
        else None
    )

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds is not None else None
    )
    cross_loader = (
        DataLoader(cross_ds, batch_size=batch_size, shuffle=False) if cross_ds is not None else None
    )
    return train_loader, val_loader, test_loader, cross_loader


def train_single_config(
    df: Optional[pd.DataFrame] = None,
    features_dir: Path = Path("features"),
    num_frames: int = 16,
    rnn_type: str = "lstm",
    hidden_size: int = 256,
    num_layers: int = 1,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 30,
    patience: int = 7,
    threshold: float = 0.5,
    checkpoint_path: Path = Path("baseline.pt"),
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    use_augmented_train: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, List[float]], Dict[str, pd.DataFrame]]:
    """
    Huấn luyện 1 cấu hình mô hình Baseline.
    Chọn checkpoint tốt nhất CHỈ dựa trên tập Validation RLVS (ưu tiên val_f1 và val_loss).
    Trả về (metrics_summary, history, preds_by_split).
    """
    set_seed(seed)

    train_loader, val_loader, test_loader, cross_loader = build_dataloaders_for_config(
        features_dir=features_dir,
        fallback_df=df,
        num_frames=num_frames,
        batch_size=batch_size,
        use_augmented_train=use_augmented_train,
    )

    model = ViolenceBaselineRNN(
        input_dim=1280,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_val_metrics: Dict[str, Any] = {}
    epochs_no_improve = 0
    best_epoch = 1

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_recall": [],
        "best_epoch": [1],
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_samples = 0

        for x, y, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            probs = model(x)
            probs_clamped = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
            loss = criterion(probs_clamped, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds_bin = (probs.detach() >= threshold).float()
            correct_train += int((preds_bin == y).sum().item())
            total_samples += x.size(0)

        train_loss = running_loss / max(total_samples, 1)
        train_acc = correct_train / max(total_samples, 1)

        val_loss, val_metrics, _ = evaluate_loader(
            model, val_loader, criterion, device, threshold=threshold
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_recall"].append(val_metrics["recall_violence"])

        # Lưu checkpoint tốt nhất dựa trên tập val (giảm val_loss hoặc tăng val_f1 rõ rệt)
        improved_loss = val_loss < best_val_loss - 1e-4
        improved_f1 = (val_metrics["f1"] > best_val_f1 + 1e-3) and (val_loss <= best_val_loss + 0.03)

        if improved_loss or improved_f1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            history["best_epoch"] = [float(best_epoch)]
            best_val_metrics = {
                "val_loss": float(val_loss),
                **val_metrics,
                "best_epoch": int(epoch),
            }
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": 1280,
                        "num_frames": num_frames,
                        "rnn_type": rnn_type,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "threshold": threshold,
                    },
                    "best_threshold": threshold,
                    "val_metrics": best_val_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"  [Early Stopping] Dừng sớm tại epoch {epoch} "
                    f"(checkpoint tốt nhất tại epoch {best_epoch})."
                )
                break

    # Tải lại checkpoint tốt nhất để đánh giá trên val, test và cross_test
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss_final, final_val_metrics, val_preds_df = evaluate_loader(
        model, val_loader, criterion, device, threshold=threshold
    )
    opt_th, opt_val_metrics = find_optimal_threshold_on_val(
        val_preds_df["y_true"].to_numpy(),
        val_preds_df["y_prob"].to_numpy(),
    )
    best_val_metrics.update(
        {
            "val_loss": float(val_loss_final),
            **final_val_metrics,
            "optimal_val_threshold": opt_th,
            "optimal_val_f1": opt_val_metrics["f1"],
            "optimal_val_recall": opt_val_metrics["recall_violence"],
        }
    )

    preds_by_split: Dict[str, pd.DataFrame] = {"val": val_preds_df}

    if test_loader is not None:
        test_loss, test_metrics, test_preds_df = evaluate_loader(
            model, test_loader, criterion, device, threshold=threshold
        )
        preds_by_split["test"] = test_preds_df
        best_val_metrics["test_loss"] = float(test_loss)
        for k, v in test_metrics.items():
            best_val_metrics[f"test_{k}"] = v

    if cross_loader is not None:
        cross_loss, cross_metrics, cross_preds_df = evaluate_loader(
            model, cross_loader, criterion, device, threshold=threshold
        )
        preds_by_split["cross_test"] = cross_preds_df
        best_val_metrics["cross_test_loss"] = float(cross_loss)
        for k, v in cross_metrics.items():
            best_val_metrics[f"cross_test_{k}"] = v

    # Cập nhật lại checkpoint với đầy đủ thông tin val/test/cross_test metrics
    checkpoint["val_metrics"] = {
        k: v for k, v in best_val_metrics.items() if not k.startswith(("test_", "cross_test_"))
    }
    checkpoint["test_metrics"] = {
        k.replace("test_", ""): v for k, v in best_val_metrics.items() if k.startswith("test_")
    }
    checkpoint["cross_test_metrics"] = {
        k.replace("cross_test_", ""): v
        for k, v in best_val_metrics.items()
        if k.startswith("cross_test_")
    }
    torch.save(checkpoint, checkpoint_path)

    return best_val_metrics, history, preds_by_split


def run_ablation_study(
    features_dir: Path,
    fallback_df: Optional[pd.DataFrame],
    output_dir: Path,
    root_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    only_t16: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Thực hiện T10: So sánh 10 cấu hình biến thể (số frame T in {8, 16, 20},
    LSTM vs GRU, hidden_size in {128, 256}, num_layers in {1, 2}).
    Chọn cấu hình tốt nhất dựa hoàn toàn trên tập Validation RLVS.
    """
    configs = [
        {"name": "C01_LSTM_T8_H128_L1", "num_frames": 8, "rnn_type": "lstm", "hidden_size": 128, "num_layers": 1},
        {"name": "C02_LSTM_T8_H256_L1", "num_frames": 8, "rnn_type": "lstm", "hidden_size": 256, "num_layers": 1},
        {"name": "C03_GRU_T8_H256_L1", "num_frames": 8, "rnn_type": "gru", "hidden_size": 256, "num_layers": 1},
        {"name": "C04_LSTM_T16_H128_L1", "num_frames": 16, "rnn_type": "lstm", "hidden_size": 128, "num_layers": 1},
        {"name": "C05_GRU_T16_H128_L1", "num_frames": 16, "rnn_type": "gru", "hidden_size": 128, "num_layers": 1},
        {"name": "C06_LSTM_T16_H256_L1", "num_frames": 16, "rnn_type": "lstm", "hidden_size": 256, "num_layers": 1},
        {"name": "C07_GRU_T16_H256_L1", "num_frames": 16, "rnn_type": "gru", "hidden_size": 256, "num_layers": 1},
        {"name": "C08_LSTM_T16_H256_L2", "num_frames": 16, "rnn_type": "lstm", "hidden_size": 256, "num_layers": 2},
        {"name": "C09_LSTM_T20_H256_L1", "num_frames": 20, "rnn_type": "lstm", "hidden_size": 256, "num_layers": 1},
        {"name": "C10_GRU_T20_H256_L2", "num_frames": 20, "rnn_type": "gru", "hidden_size": 256, "num_layers": 2},
    ]
    if only_t16:
        configs = [c for c in configs if c["num_frames"] == 16]

    results = []
    ablation_ckpt_dir = output_dir / "ablation_checkpoints"
    ablation_ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_cfg_info: Optional[Dict[str, Any]] = None
    best_cfg_score = (-1.0, -1.0, -1.0)  # (val_f1, val_recall, val_roc_auc)

    for cfg in configs:
        print(f"\n[ABLATION T10] Đang huấn luyện cấu hình: {cfg['name']} ...")
        ckpt_path = ablation_ckpt_dir / f"{cfg['name']}.pt"
        metrics, history, preds_by_split = train_single_config(
            df=fallback_df,
            features_dir=features_dir,
            num_frames=cfg["num_frames"],
            rnn_type=cfg["rnn_type"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            checkpoint_path=ckpt_path,
            device=device,
        )

        model_label = (
            f"{cfg['rnn_type'].upper()}"
            if cfg["num_layers"] == 1
            else f"{cfg['rnn_type'].upper()} ({cfg['num_layers']} lớp)"
        )

        row = {
            "Mô hình (LSTM/GRU)": model_label,
            "Số frame (T)": cfg["num_frames"],
            "Hidden size": cfg["hidden_size"],
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall (Violence)": round(metrics["recall_violence"], 4),
            "F1": round(metrics["f1"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "Số lớp (Layers)": cfg["num_layers"],
            "Val Loss": round(metrics["val_loss"], 4),
            "Test Accuracy": round(metrics.get("test_accuracy", 0.0), 4),
            "Test Precision": round(metrics.get("test_precision", 0.0), 4),
            "Test Recall (Violence)": round(metrics.get("test_recall_violence", 0.0), 4),
            "Test F1": round(metrics.get("test_f1", 0.0), 4),
            "Test ROC-AUC": round(metrics.get("test_roc_auc", 0.0), 4),
            "Cross-Test Accuracy": round(metrics.get("cross_test_accuracy", 0.0), 4),
            "Cross-Test Precision": round(metrics.get("cross_test_precision", 0.0), 4),
            "Cross-Test Recall (Violence)": round(metrics.get("cross_test_recall_violence", 0.0), 4),
            "Cross-Test F1": round(metrics.get("cross_test_f1", 0.0), 4),
            "Cross-Test ROC-AUC": round(metrics.get("cross_test_roc_auc", 0.0), 4),
            "Mã cấu hình": cfg["name"],
        }
        results.append(row)

        # Ưu tiên chọn checkpoint chính có T=16 cho Demo T22 (hoặc cấu hình T=16 đạt F1 cao nhất trên val)
        score_tuple = (
            1 if cfg["num_frames"] == 16 else 0,
            metrics["f1"],
            metrics["recall_violence"],
            metrics["roc_auc"],
        )
        if best_cfg_info is None or score_tuple > best_cfg_score:
            best_cfg_score = score_tuple
            best_cfg_info = {
                "cfg": cfg,
                "ckpt_path": ckpt_path,
                "metrics": metrics,
                "history": history,
                "preds_by_split": preds_by_split,
            }

    ablation_df = pd.DataFrame(results)
    ablation_paths = [
        output_dir / "ablation_baseline.csv",
        root_dir / "ablation_baseline.csv",
    ]
    for p in ablation_paths:
        ablation_df.to_csv(p, index=False, encoding="utf-8-sig")

    plot_ablation_comparison(
        ablation_df,
        save_paths=[
            output_dir / "ablation_comparison.png",
            root_dir / "ablation_comparison.png",
        ],
    )

    print(f"\n[HOÀN TẤT T10] Đã lưu bảng so sánh {len(ablation_df)} cấu hình tại: {ablation_paths[1]}")
    print(
        ablation_df[
            [
                "Mô hình (LSTM/GRU)",
                "Số frame (T)",
                "Hidden size",
                "Accuracy",
                "Precision",
                "Recall (Violence)",
                "F1",
                "ROC-AUC",
            ]
        ].to_string(index=False)
    )
    assert best_cfg_info is not None
    return ablation_df, best_cfg_info


def run_t14_augmentation_comparison(
    features_dir: Path,
    fallback_df: Optional[pd.DataFrame],
    output_dir: Path,
    root_dir: Path,
    best_cfg: Dict[str, Any],
    no_aug_metrics: Dict[str, Any],
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> pd.DataFrame:
    """
    Thực hiện T14: So sánh mô hình Baseline khi huấn luyện Không có vs. Có tăng cường dữ liệu
    (augment_clip trên tập train: lật ngang, đổi độ sáng, cắt ngẫu nhiên).
    """
    print("\n[T14 AUGMENTATION] Đang huấn luyện Baseline với Tăng cường dữ liệu (augment_clip)...")
    aug_ckpt_path = output_dir / "baseline_with_aug.pt"
    aug_metrics, _, _ = train_single_config(
        df=fallback_df,
        features_dir=features_dir,
        num_frames=best_cfg["num_frames"],
        rnn_type=best_cfg["rnn_type"],
        hidden_size=best_cfg["hidden_size"],
        num_layers=best_cfg["num_layers"],
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        checkpoint_path=aug_ckpt_path,
        device=device,
        use_augmented_train=True,
    )

    rows = [
        {
            "Phương pháp huấn luyện": "Không tăng cường dữ liệu (No Augmentation)",
            "Mô hình": f"{best_cfg['rnn_type'].upper()} (T={best_cfg['num_frames']}, H={best_cfg['hidden_size']})",
            "Val Accuracy": round(no_aug_metrics["accuracy"], 4),
            "Val Precision": round(no_aug_metrics["precision"], 4),
            "Val Recall (Violence)": round(no_aug_metrics["recall_violence"], 4),
            "Val F1": round(no_aug_metrics["f1"], 4),
            "Val ROC-AUC": round(no_aug_metrics["roc_auc"], 4),
            "Test Accuracy": round(no_aug_metrics.get("test_accuracy", 0.0), 4),
            "Test Precision": round(no_aug_metrics.get("test_precision", 0.0), 4),
            "Test Recall (Violence)": round(no_aug_metrics.get("test_recall_violence", 0.0), 4),
            "Test F1": round(no_aug_metrics.get("test_f1", 0.0), 4),
            "Test ROC-AUC": round(no_aug_metrics.get("test_roc_auc", 0.0), 4),
            "Cross-Test Accuracy": round(no_aug_metrics.get("cross_test_accuracy", 0.0), 4),
            "Cross-Test F1": round(no_aug_metrics.get("cross_test_f1", 0.0), 4),
            "Cross-Test ROC-AUC": round(no_aug_metrics.get("cross_test_roc_auc", 0.0), 4),
        },
        {
            "Phương pháp huấn luyện": "Có tăng cường dữ liệu (With augment_clip)",
            "Mô hình": f"{best_cfg['rnn_type'].upper()} (T={best_cfg['num_frames']}, H={best_cfg['hidden_size']})",
            "Val Accuracy": round(aug_metrics["accuracy"], 4),
            "Val Precision": round(aug_metrics["precision"], 4),
            "Val Recall (Violence)": round(aug_metrics["recall_violence"], 4),
            "Val F1": round(aug_metrics["f1"], 4),
            "Val ROC-AUC": round(aug_metrics["roc_auc"], 4),
            "Test Accuracy": round(aug_metrics.get("test_accuracy", 0.0), 4),
            "Test Precision": round(aug_metrics.get("test_precision", 0.0), 4),
            "Test Recall (Violence)": round(aug_metrics.get("test_recall_violence", 0.0), 4),
            "Test F1": round(aug_metrics.get("test_f1", 0.0), 4),
            "Test ROC-AUC": round(aug_metrics.get("test_roc_auc", 0.0), 4),
            "Cross-Test Accuracy": round(aug_metrics.get("cross_test_accuracy", 0.0), 4),
            "Cross-Test F1": round(aug_metrics.get("cross_test_f1", 0.0), 4),
            "Cross-Test ROC-AUC": round(aug_metrics.get("cross_test_roc_auc", 0.0), 4),
        },
    ]

    t14_df = pd.DataFrame(rows)
    for p in [
        output_dir / "t14_augmentation_baseline.csv",
        root_dir / "t14_augmentation_baseline.csv",
    ]:
        t14_df.to_csv(p, index=False, encoding="utf-8-sig")

    print("\n[HOÀN TẤT T14] Bảng đối chiếu hiệu quả Tăng cường dữ liệu (Data Augmentation):")
    print(t14_df.to_string(index=False))
    return t14_df


def save_all_predictions_and_plots(
    preds_by_split: Dict[str, pd.DataFrame],
    history: Dict[str, List[float]],
    root_dir: Path,
    output_dir: Path,
) -> None:
    """
    Lưu toàn bộ file dự đoán theo đúng quy ước với Thiên (Mã D):
      - preds_baseline/preds_baseline_val.csv
      - preds_baseline/preds_baseline_test.csv
      - preds_baseline/preds_baseline.csv (val + test)
      - preds_baseline/preds_baseline_cross_test.csv (RWF-2000)
      - Bản sao preds_baseline.csv tại thư mục gốc và outputs_baseline/
    Và vẽ các biểu đồ đường cong huấn luyện, ma trận nhầm lẫn, đường cong ROC.
    """
    preds_folder = root_dir / "preds_baseline"
    preds_folder.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_df = preds_by_split["val"]
    val_df.to_csv(preds_folder / "preds_baseline_val.csv", index=False)

    combined_parts = [val_df]
    if "test" in preds_by_split:
        test_df = preds_by_split["test"]
        test_df.to_csv(preds_folder / "preds_baseline_test.csv", index=False)
        combined_parts.append(test_df)

    combined_df = pd.concat(combined_parts, ignore_index=True)
    combined_df.to_csv(preds_folder / "preds_baseline.csv", index=False)
    combined_df.to_csv(root_dir / "preds_baseline.csv", index=False)
    combined_df.to_csv(output_dir / "preds_baseline.csv", index=False)

    if "cross_test" in preds_by_split:
        cross_df = preds_by_split["cross_test"]
        cross_df.to_csv(preds_folder / "preds_baseline_cross_test.csv", index=False)
        cross_df.to_csv(output_dir / "preds_baseline_cross_test.csv", index=False)

    # Vẽ biểu đồ đường cong huấn luyện
    plot_training_curves(
        history,
        save_paths=[root_dir / "loss_curve.png", output_dir / "loss_curve.png"],
    )

    # Vẽ Ma trận nhầm lẫn & Đường cong ROC
    plot_splits: Dict[str, pd.DataFrame] = {
        f"Tập Validation RLVS (N={len(val_df)})": val_df,
    }
    if "test" in preds_by_split:
        plot_splits[f"Tập Test RLVS (N={len(preds_by_split['test'])})"] = preds_by_split["test"]
    if "cross_test" in preds_by_split:
        plot_splits[f"Tập Cross-test RWF-2000 (N={len(preds_by_split['cross_test'])})"] = (
            preds_by_split["cross_test"]
        )

    plot_confusion_matrices(
        plot_splits,
        save_paths=[
            root_dir / "confusion_matrix_baseline.png",
            output_dir / "confusion_matrix_baseline.png",
        ],
    )
    plot_roc_curves(
        plot_splits,
        save_paths=[
            root_dir / "roc_curve_baseline.png",
            output_dir / "roc_curve_baseline.png",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T09, T10, T14, T19, T20: Huấn luyện và đánh giá toàn diện Baseline CNN + LSTM/GRU"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="features",
        help="Thư mục chứa đặc trưng MobileNetV2 từ T08",
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default="features/features_T16.csv",
        help="File CSV chỉ mục đặc trưng dự phòng từ T08",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Số lượng frame mỗi video mặc định (8, 16 hoặc 20)",
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="lstm",
        choices=["lstm", "gru"],
        help="Loại mạng tuần tự mặc định: lstm hoặc gru",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Kích thước vector trạng thái ẩn của LSTM/GRU",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        choices=[1, 2],
        help="Số lớp LSTM/GRU (1 hoặc 2)",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Tỷ lệ Dropout")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate cho Adam")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=35, help="Số epoch tối đa")
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Số epoch chờ Early Stopping",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_baseline",
        help="Thư mục lưu kết quả tổng hợp",
    )
    parser.add_argument(
        "--run_ablation",
        action="store_true",
        default=True,
        help="Chạy thí nghiệm biến thể T10 và so sánh tăng cường dữ liệu T14",
    )
    parser.add_argument(
        "--only_t16",
        action="store_true",
        default=False,
        help="Chỉ chạy trước các cấu hình T=16 để xuất ngay baseline.pt và preds_baseline/",
    )
    args = parser.parse_args()

    root_dir = Path(".").resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir).resolve()

    fallback_df: Optional[pd.DataFrame] = None
    features_csv_path = Path(args.features_csv)
    if features_csv_path.exists():
        fallback_df = pd.read_csv(features_csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị huấn luyện: {device}")

    root_ckpt_path = root_dir / "baseline.pt"
    out_ckpt_path = output_dir / "baseline.pt"

    if args.run_ablation:
        ablation_df, best_cfg_info = run_ablation_study(
            features_dir=features_dir,
            fallback_df=fallback_df,
            output_dir=output_dir,
            root_dir=root_dir,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            only_t16=args.only_t16,
        )
        best_cfg = best_cfg_info["cfg"]
        best_metrics = best_cfg_info["metrics"]
        history = best_cfg_info["history"]
        preds_by_split = best_cfg_info["preds_by_split"]

        # Sao chép checkpoint tốt nhất trên tập Val (T=16) ra baseline.pt
        shutil.copy2(best_cfg_info["ckpt_path"], root_ckpt_path)
        shutil.copy2(best_cfg_info["ckpt_path"], out_ckpt_path)
        print(
            f"\n[CHỌN MÔ HÌNH TỐT NHẤT TRÊN VAL] Cấu hình: {best_cfg['name']} "
            f"(RNN={best_cfg['rnn_type'].upper()}, T={best_cfg['num_frames']}, "
            f"H={best_cfg['hidden_size']}, Layers={best_cfg['num_layers']})"
        )

        # Chạy đối chiếu Tăng cường dữ liệu T14 trên cấu hình tốt nhất
        t14_df = run_t14_augmentation_comparison(
            features_dir=features_dir,
            fallback_df=fallback_df,
            output_dir=output_dir,
            root_dir=root_dir,
            best_cfg=best_cfg,
            no_aug_metrics=best_metrics,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
    else:
        best_metrics, history, preds_by_split = train_single_config(
            df=fallback_df,
            features_dir=features_dir,
            num_frames=args.num_frames,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            checkpoint_path=root_ckpt_path,
            device=device,
        )
        shutil.copy2(root_ckpt_path, out_ckpt_path)
        best_cfg = {
            "name": f"{args.rnn_type.upper()}_T{args.num_frames}_H{args.hidden_size}_L{args.num_layers}",
            "num_frames": args.num_frames,
            "rnn_type": args.rnn_type,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
        }
        ablation_df = pd.DataFrame()
        t14_df = pd.DataFrame()

    # Lưu tất cả file dự đoán và biểu đồ
    save_all_predictions_and_plots(
        preds_by_split=preds_by_split,
        history=history,
        root_dir=root_dir,
        output_dir=output_dir,
    )

    # Lưu tổng hợp JSON để phục vụ báo cáo T24 và kiểm tra tự động
    summary_payload = {
        "best_config": best_cfg,
        "metrics": best_metrics,
        "ablation_rows": ablation_df.to_dict(orient="records") if not ablation_df.empty else [],
        "t14_augmentation_rows": t14_df.to_dict(orient="records") if not t14_df.empty else [],
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("[HOÀN TẤT TOÀN BỘ T09, T10, T14, T19, T20] TỔNG HỢP KẾT QUẢ CHÍNH:")
    print(f"  - Cấu hình chọn từ Val: {best_cfg['name']}")
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.4f}")
        else:
            print(f"  - {k}: {v}")
    print(f"  - Checkpoint chính: {root_ckpt_path}")
    print(f"  - Thư mục dự đoán: {root_dir / 'preds_baseline'}")
    print("=" * 72)


if __name__ == "__main__":
    main()