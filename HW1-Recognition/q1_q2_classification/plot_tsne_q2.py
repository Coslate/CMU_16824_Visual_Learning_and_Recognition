import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from voc_dataset import VOCDataset
from train_q2 import ResNet   # 你的 ResNet class


def get_class_colors(num_classes=20):
    """
    return 20 colors, shape = [20, 3]
    """
    cmap = plt.get_cmap("tab20")
    colors = np.array([cmap(i)[:3] for i in range(num_classes)], dtype=np.float32)
    return colors


def label_to_color(label_vec, class_colors):
    """
    label_vec: shape [20], 0/1
    return average of active classes color if multilabels
    """
    active = np.where(label_vec > 0)[0]
    if len(active) == 0:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)  # fallback gray color
    return class_colors[active].mean(axis=0)


@torch.no_grad()
def extract_features_and_labels(model, loader, device):
    model.eval()

    feat_list = []
    label_list = []

    for images, labels, wgts in loader:
        images = images.to(device)

        # use penultimate feature, not logits
        feats = model.forward_features(images)   # [B, 512]

        feat_list.append(feats.cpu())
        label_list.append(labels.cpu())

    feats = torch.cat(feat_list, dim=0).numpy()     # [N, 512]
    labels = torch.cat(label_list, dim=0).numpy()   # [N, 20]
    return feats, labels


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1. Build dataset/test loader
    # --------------------------------------------------
    dataset = VOCDataset(split="test", size=224)

    num_samples = 1000
    assert len(dataset) >= num_samples, "test set 少於 1000 張，請檢查資料集"

    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )

    # --------------------------------------------------
    # 2. Build model and load in fine-tuned weights
    # --------------------------------------------------
    model = ResNet(num_classes=len(VOCDataset.CLASS_NAMES)).to(device)
    ckpt_path = "checkpoint-model-epoch50.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --------------------------------------------------
    # 3. Extract feature
    # --------------------------------------------------
    feats, labels = extract_features_and_labels(model, loader, device)
    print("features shape:", feats.shape)   # (1000, 512)
    print("labels shape:", labels.shape)    # (1000, 20)

    # --------------------------------------------------
    # 4. t-SNE dimension reduction
    # --------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    feats_2d = tsne.fit_transform(feats)   # [1000, 2]

    # --------------------------------------------------
    # 5. Produce color for each point
    # --------------------------------------------------
    class_colors = get_class_colors(len(VOCDataset.CLASS_NAMES))
    point_colors = np.array([label_to_color(labels[i], class_colors) for i in range(len(labels))])

    # --------------------------------------------------
    # 6. Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 8))
    plt.scatter(
        feats_2d[:, 0],
        feats_2d[:, 1],
        c=point_colors,
        s=18,
        alpha=0.8
    )

    plt.title("t-SNE of Fine-tuned ResNet-18 Features on PASCAL VOC Test Set")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # legend: class color indication
    legend_handles = []
    for i, name in enumerate(VOCDataset.CLASS_NAMES):
        legend_handles.append(Patch(facecolor=class_colors[i], edgecolor='black', label=name))

    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig("./fig/Q2_6_tsne.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()