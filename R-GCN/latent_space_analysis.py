from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "R-GCN" / "rgcn_model.py").exists():
            return candidate
    raise RuntimeError("Could not find project root containing R-GCN/rgcn_model.py")


RGCN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(RGCN_DIR)
if str(RGCN_DIR) not in sys.path:
    sys.path.insert(0, str(RGCN_DIR))
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mpl").resolve()))

import matplotlib.pyplot as plt

from rgcn_model import RGCNLinkPredictor
from vgae_model import VGAE


DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


class RelationalGraphPlaceholder:
    pass


class GraphCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "RelationalGraph":
            return RelationalGraphPlaceholder
        return super().find_class(module, name)


@dataclass
class LatentArtifacts:
    graph: object
    label_to_rel: dict[str, int]
    node_to_qid: dict[int, str]
    human_node_ids: np.ndarray
    rgcn_latents: np.ndarray
    vgae_mu: np.ndarray
    vgae_std: np.ndarray
    hidden_dim: int
    latent_dim: int
    graph_checkpoint: Path
    rgcn_checkpoint: Path
    vgae_results_path: Path


def _edge_arrays_for_splits(graph, split_names: tuple[str, ...]) -> np.ndarray:
    arrays = []
    for split_name in split_names:
        edge_tensor = getattr(graph, f"{split_name}_edges")
        arrays.append(edge_tensor.cpu().numpy())
    return np.concatenate(arrays, axis=0)


def occupation_family_for_label(label: str) -> str:
    value = str(label).strip().lower()

    if value in {"pornographic actor", "playboy playmate"}:
        return "excluded"

    if any(token in value for token in ["priest", "bishop", "parson", "missionary", "prelate", "mormon"]):
        return "religious"
    if any(token in value for token in ["judge", "lawyer", "jurist", "police", "sheriff", "counsel", "executioner"]):
        return "law_judiciary"
    if any(token in value for token in ["military", "fighter pilot", "intelligence agent"]):
        return "military"
    if any(token in value for token in ["football", "coach", "referee", "bullfighter", "professional wrestling"]):
        return "sports"
    if any(
        token in value
        for token in [
            "actor",
            "playmate",
            "presenter",
            "animator",
            "idol",
            "singer",
            "musician",
            "composer",
            "reporter",
            "editor",
            "illustrator",
            "artist",
            "organist",
        ]
    ):
        return "entertainment_media"
    if any(token in value for token in ["teacher", "professor", "instructor", "scholar", "docent", "educator", "teaching", "librarian"]):
        return "academia_education"
    if any(
        token in value
        for token in [
            "politician",
            "diplomat",
            "ambassador",
            "governor",
            "chancellor",
            "civil servant",
            "chairperson",
            "secretary",
            "undersecretary",
            "official",
            "municipal clerk",
        ]
    ):
        return "politics_public_service"
    if any(
        token in value
        for token in [
            "entrepreneur",
            "businessperson",
            "banker",
            "insurance broker",
            "claims adjuster",
            "bank teller",
            "manager",
            "executive director",
            "ship-owner",
            "middle management",
        ]
    ):
        return "business_management"
    if any(token in value for token in ["physician", "nurse", "personal care assistant"]):
        return "medicine_care"
    if any(token in value for token in ["astronaut", "engineer", "inventor", "expert"]):
        return "science_technology"
    if any(token in value for token in ["translator", "bookseller", "curator", "gardener", "miner", "welder"]):
        return "skilled_professions"
    return "other"


def load_graph(graph_checkpoint: Path):
    with graph_checkpoint.open("rb") as f:
        return GraphCheckpointUnpickler(f).load()


def load_human_node_ids(human_cache_path: Path) -> np.ndarray:
    with human_cache_path.open("rb") as f:
        human_cache = pickle.load(f)
    return np.asarray(human_cache["human_node_ids"], dtype=np.int64)


def load_node_maps(node_id_map_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with node_id_map_path.open("rb") as f:
        qid_to_node = pickle.load(f)
    node_to_qid = {int(node_id): qid for qid, node_id in qid_to_node.items()}
    return qid_to_node, node_to_qid


def compute_latents(
    graph,
    rgcn_checkpoint: Path,
    vgae_results_path: Path,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    with vgae_results_path.open("rb") as f:
        vgae_results = pickle.load(f)

    vgae_config = vgae_results["config"]
    hidden_dim = int(vgae_config.get("hidden_dim", 64))
    latent_dim = int(vgae_config.get("latent_dim", hidden_dim))
    feat_dim = graph.node_features.shape[1] if graph.node_features is not None else None

    rgcn_model = RGCNLinkPredictor(
        num_nodes=graph.num_nodes,
        num_relations=graph.num_relations,
        hidden_dim=hidden_dim,
    ).to(device)
    rgcn_model.load_state_dict(torch.load(rgcn_checkpoint, map_location=device))
    rgcn_model.eval()

    vgae_model = VGAE(
        num_nodes=graph.num_nodes,
        hidden_dim=hidden_dim,
        num_relations=graph.num_relations,
        latent_dim=latent_dim,
        feat_dim=feat_dim,
    ).to(device)
    vgae_model.load_state_dict(vgae_results["model_state_dict"])
    vgae_model.eval()

    train_edges = graph.train_edges.to(device)
    edge_index = train_edges[:, :2].t().contiguous()
    edge_type = train_edges[:, 2]
    node_features = graph.node_features.to(device) if graph.node_features is not None else None

    with torch.no_grad():
        rgcn_latents = rgcn_model.encoder(
            edge_index,
            edge_type,
            node_features=node_features,
            num_nodes=graph.num_nodes,
        ).detach().cpu().numpy()
        _, vgae_mu, vgae_log_var = vgae_model(
            edge_index,
            edge_type,
            node_features=node_features,
            num_nodes=graph.num_nodes,
        )
        vgae_mu = vgae_mu.detach().cpu().numpy()
        vgae_std = torch.exp(0.5 * vgae_log_var).detach().cpu().numpy()

    return rgcn_latents, vgae_mu, vgae_std, hidden_dim, latent_dim


def load_latent_artifacts(project_root: Path | None = None) -> LatentArtifacts:
    project_root = PROJECT_ROOT if project_root is None else find_project_root(project_root)
    rgcn_dir = project_root / "R-GCN"
    checkpoint_dir = rgcn_dir / "checkpoints"

    graph_checkpoint = checkpoint_dir / "wiki_graph.pkl"
    rgcn_checkpoint = checkpoint_dir / "rgcn_scratch_wikidata.pt"
    vgae_results_path = checkpoint_dir / "vgae_wikidata_results.pkl"
    node_id_map_path = rgcn_dir / "datasets" / "tkgl_smallpedia" / "ml_tkgl-smallpedia_nodeid.pkl"
    human_cache_path = checkpoint_dir / "wikidata_human_nodes.pkl"

    graph = load_graph(graph_checkpoint)
    _, node_to_qid = load_node_maps(node_id_map_path)
    human_node_ids = load_human_node_ids(human_cache_path)
    rgcn_latents, vgae_mu, vgae_std, hidden_dim, latent_dim = compute_latents(
        graph=graph,
        rgcn_checkpoint=rgcn_checkpoint,
        vgae_results_path=vgae_results_path,
        device=DEVICE,
    )

    return LatentArtifacts(
        graph=graph,
        label_to_rel={label: rel_id for rel_id, label in graph.rel_labels.items()},
        node_to_qid=node_to_qid,
        human_node_ids=human_node_ids,
        rgcn_latents=rgcn_latents,
        vgae_mu=vgae_mu,
        vgae_std=vgae_std,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        graph_checkpoint=graph_checkpoint,
        rgcn_checkpoint=rgcn_checkpoint,
        vgae_results_path=vgae_results_path,
    )


def _relation_edge_frame(
    graph,
    subject_node_ids: np.ndarray,
    relation_id: int,
    *,
    split_names: tuple[str, ...] = ("train", "val", "test"),
) -> pd.DataFrame:
    edges = _edge_arrays_for_splits(graph, split_names)
    src = edges[:, 0]
    dst = edges[:, 1]
    rel = edges[:, 2]

    subject_mask = np.zeros(graph.num_nodes, dtype=bool)
    subject_mask[np.asarray(subject_node_ids, dtype=np.int64)] = True

    mask = (rel == relation_id) & subject_mask[src]
    rows = pd.DataFrame({"node_id": src[mask], "target_node_id": dst[mask]})
    rows = rows.drop_duplicates()
    rows["target_label"] = rows["target_node_id"].map(
        lambda node_id: graph.node_labels.get(int(node_id), str(int(node_id)))
    )
    rows["target_label"] = rows["target_label"].astype(str).str.strip()
    return rows[rows["target_label"] != ""].copy()


def build_semantic_category_frame(
    graph,
    subject_node_ids: np.ndarray,
    relation_label: str,
    label_to_rel: dict[str, int],
    *,
    top_k: int = 8,
    min_category_size: int = 10,
    max_nodes_per_category: int = 250,
    seed: int = 42,
    split_names: tuple[str, ...] = ("train", "val", "test"),
    collapse_occupation_families: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation_id = label_to_rel[relation_label]
    relation_rows = _relation_edge_frame(
        graph,
        subject_node_ids,
        relation_id,
        split_names=split_names,
    ).rename(columns={"target_label": "raw_category"})

    if collapse_occupation_families and relation_label == "occupation":
        relation_rows["category"] = relation_rows["raw_category"].map(occupation_family_for_label)
        relation_rows = relation_rows[relation_rows["category"] != "excluded"].copy()
    else:
        relation_rows["category"] = relation_rows["raw_category"]

    category_sizes = (
        relation_rows.groupby("category")["node_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("num_nodes")
        .reset_index()
    )
    category_sizes = category_sizes[category_sizes["num_nodes"] >= min_category_size].copy()
    category_sizes = category_sizes.head(top_k).reset_index(drop=True)

    if category_sizes.empty:
        raise ValueError(
            f"No categories for relation '{relation_label}' met min_category_size={min_category_size}."
        )

    category_order = category_sizes["category"].tolist()
    category_rank = {category: rank for rank, category in enumerate(category_order)}
    raw_category_rank = {
        raw_category: rank
        for rank, raw_category in enumerate(
            relation_rows.groupby("raw_category")["node_id"]
            .nunique()
            .sort_values(ascending=False)
            .index
        )
    }

    assigned = relation_rows[relation_rows["category"].isin(category_rank)].copy()
    assigned["category_rank"] = assigned["category"].map(category_rank)
    assigned["raw_category_rank"] = assigned["raw_category"].map(raw_category_rank)
    assigned = assigned.sort_values(["node_id", "category_rank", "raw_category_rank"])
    assigned = assigned.drop_duplicates(subset=["node_id"], keep="first")

    node_frame = pd.DataFrame({"node_id": np.asarray(subject_node_ids, dtype=np.int64)})
    node_frame = node_frame.merge(
        assigned[["node_id", "category", "raw_category"]],
        on="node_id",
        how="inner",
    )
    node_frame["label"] = node_frame["node_id"].map(
        lambda node_id: graph.node_labels.get(int(node_id), str(int(node_id)))
    )

    if max_nodes_per_category is not None:
        sampled_parts = []
        for _, frame in node_frame.groupby("category", sort=False):
            sampled_parts.append(
                frame.sample(
                    n=min(len(frame), max_nodes_per_category),
                    random_state=seed,
                )
            )
        sampled = pd.concat(sampled_parts, ignore_index=True)
    else:
        sampled = node_frame.reset_index(drop=True)

    sampled["category"] = pd.Categorical(sampled["category"], categories=category_order, ordered=True)
    sampled = sampled.sort_values(["category", "label", "node_id"]).reset_index(drop=True)

    sampled_counts = (
        sampled.groupby("category", observed=True)["node_id"]
        .count()
        .rename("sampled_nodes")
        .reset_index()
    )
    category_examples = (
        relation_rows.groupby(["category", "raw_category"])["node_id"]
        .nunique()
        .rename("count")
        .reset_index()
        .sort_values(["category", "count", "raw_category"], ascending=[True, False, True])
        .groupby("category", sort=False)
        .head(4)
        .groupby("category", sort=False)
        .apply(
            lambda frame: ", ".join(
                f"{row.raw_category} ({int(row.count)})"
                for row in frame.itertuples(index=False)
            )
        )
        .rename("top_labels")
        .reset_index()
    )
    summary = category_sizes.merge(sampled_counts, on="category", how="left")
    summary = summary.merge(category_examples, on="category", how="left")
    summary["sampled_nodes"] = summary["sampled_nodes"].fillna(0).astype(int)

    return sampled, summary


def project_latents_with_pca(
    latent_matrix: np.ndarray,
    node_frame: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, PCA]:
    node_ids = node_frame["node_id"].to_numpy(dtype=np.int64)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(latent_matrix[node_ids])

    projection = node_frame.copy()
    projection["model"] = model_name
    projection["pc1"] = coords[:, 0]
    projection["pc2"] = coords[:, 1]
    projection["explained_variance_ratio"] = float(pca.explained_variance_ratio_.sum())
    return projection, pca


def add_kmeans_clusters(
    projection: pd.DataFrame,
    latent_matrix: np.ndarray,
    *,
    cluster_col: str = "cluster",
    random_state: int = 42,
) -> pd.DataFrame:
    node_ids = projection["node_id"].to_numpy(dtype=np.int64)
    x = StandardScaler().fit_transform(latent_matrix[node_ids])
    num_clusters = projection["category"].nunique()
    kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=random_state)
    clustered = projection.copy()
    clustered[cluster_col] = pd.Categorical(
        [f"Cluster {cluster_id + 1}" for cluster_id in kmeans.fit_predict(x)]
    )
    return clustered


def _cluster_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    contingency = pd.crosstab(cluster_labels, true_labels)
    return float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())


def compute_smoothness_metrics(
    latent_matrix: np.ndarray,
    node_frame: pd.DataFrame,
    *,
    neighbors: int = 15,
    random_state: int = 42,
) -> dict[str, float]:
    node_ids = node_frame["node_id"].to_numpy(dtype=np.int64)
    x = StandardScaler().fit_transform(latent_matrix[node_ids])
    labels = node_frame["category"].astype(str).to_numpy()

    effective_neighbors = min(neighbors + 1, len(node_frame))
    nn = NearestNeighbors(n_neighbors=effective_neighbors)
    nn.fit(x)
    _, neighbor_idx = nn.kneighbors(x)
    neighbor_idx = neighbor_idx[:, 1:]
    neighbor_labels = labels[neighbor_idx]
    local_agreement = (neighbor_labels == labels[:, None]).mean(axis=1)

    kmeans = KMeans(
        n_clusters=node_frame["category"].nunique(),
        n_init=20,
        random_state=random_state,
    )
    cluster_labels = kmeans.fit_predict(x)

    return {
        "nodes": float(len(node_frame)),
        "categories": float(node_frame["category"].nunique()),
        "knn_label_agreement_mean": float(local_agreement.mean()),
        "knn_label_agreement_median": float(np.median(local_agreement)),
        "silhouette_by_category": float(silhouette_score(x, labels)),
        "cluster_nmi": float(normalized_mutual_info_score(labels, cluster_labels)),
        "cluster_ari": float(adjusted_rand_score(labels, cluster_labels)),
        "cluster_purity": float(_cluster_purity(labels, cluster_labels)),
    }


def compare_model_metrics(
    node_frame: pd.DataFrame,
    rgcn_latents: np.ndarray,
    vgae_latents: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for model_name, latents in [("R-GCN", rgcn_latents), ("VGAE", vgae_latents)]:
        metrics = compute_smoothness_metrics(latents, node_frame)
        metrics["model"] = model_name
        rows.append(metrics)

    metrics_frame = pd.DataFrame(rows)
    ordered_cols = [
        "model",
        "nodes",
        "categories",
        "knn_label_agreement_mean",
        "knn_label_agreement_median",
        "silhouette_by_category",
        "cluster_nmi",
        "cluster_ari",
        "cluster_purity",
    ]
    metrics_frame = metrics_frame[ordered_cols]
    metrics_frame["nodes"] = metrics_frame["nodes"].astype(int)
    metrics_frame["categories"] = metrics_frame["categories"].astype(int)
    return metrics_frame.round(4)


def make_category_color_map(category_values: list[str]) -> dict[str, str]:
    palette = [
        "#0f4c81",
        "#d1495b",
        "#2a9d8f",
        "#e9c46a",
        "#6d597a",
        "#8ab17d",
        "#ef476f",
        "#118ab2",
        "#bc6c25",
        "#7f5539",
        "#ff7f11",
        "#6a994e",
    ]
    return {
        category: palette[idx % len(palette)]
        for idx, category in enumerate(category_values)
    }


def plot_side_by_side(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    color_col: str,
    title: str,
    left_title: str,
    right_title: str,
    color_map: dict[str, str] | None = None,
) -> plt.Figure:
    combined_values = pd.Index(left_frame[color_col]).append(pd.Index(right_frame[color_col]))
    categories = [str(value) for value in pd.unique(combined_values.astype(str))]
    if color_map is None:
        color_map = make_category_color_map(categories)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, frame, subplot_title in zip(axes, [left_frame, right_frame], [left_title, right_title]):
        for category in categories:
            subset = frame[frame[color_col].astype(str) == category]
            if subset.empty:
                continue
            ax.scatter(
                subset["pc1"],
                subset["pc2"],
                s=22,
                alpha=0.72,
                c=color_map[category],
                label=category,
                linewidths=0,
            )
        ax.set_title(subplot_title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.18, linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title=color_col.replace("_", " ").title(),
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    return fig
