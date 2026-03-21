from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
import torchmetrics as tm
from torchmetrics import retrieval

from final_project.src.config import CFG


import logging

logger = logging.getLogger(__name__)


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ]
        )
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class TwoTowerModel(LightningModule):
    def __init__(
        self,
        config: CFG | None,
        cardinalities: dict[str, int],
        user_cat_feature_names: Sequence[str],
        item_cat_feature_names: Sequence[str],
        user_num_dim: int,
        item_num_dim: int,
        embedding_dim: int = 64,
        user_hidden_dims: Sequence[int] = (128,),
        item_hidden_dims: Sequence[int] = (128,),
        normalize_embeddings: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        ranking_ks: Sequence[int] = (5, 10),
        scheduler_name: str = "cosine",
        scheduler_t_max: int = 10,
        scheduler_eta_min: float = 1e-6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lr = config.training.lr if config is not None else lr

        self.user_cat_feature_names = list(user_cat_feature_names)
        self.item_cat_feature_names = list(item_cat_feature_names)
        self.normalize_embeddings = normalize_embeddings
        self.ranking_ks = tuple(sorted(set(int(k) for k in ranking_ks)))

        self.user_cat_embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(cardinalities[col], embedding_dim, padding_idx=0)
                for col in self.user_cat_feature_names
            }
        )
        self.item_cat_embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(cardinalities[col], embedding_dim, padding_idx=0)
                for col in self.item_cat_feature_names
            }
        )

        user_input_dim = len(self.user_cat_feature_names) * embedding_dim + int(user_num_dim)
        item_input_dim = len(self.item_cat_feature_names) * embedding_dim + int(item_num_dim)

        self.user_tower = _build_mlp(
            input_dim=user_input_dim,
            hidden_dims=user_hidden_dims,
            output_dim=embedding_dim,
        )
        self.item_tower = _build_mlp(
            input_dim=item_input_dim,
            hidden_dims=item_hidden_dims,
            output_dim=embedding_dim,
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_loss_metric = tm.MeanMetric()
        self.val_loss_metric = tm.MeanMetric()
        self.test_loss_metric = tm.MeanMetric()

        self.train_recall_metrics = nn.ModuleDict()
        self.val_recall_metrics = nn.ModuleDict()
        self.test_recall_metrics = nn.ModuleDict()
        self.train_ndcg_metrics = nn.ModuleDict()
        self.val_ndcg_metrics = nn.ModuleDict()
        self.test_ndcg_metrics = nn.ModuleDict()
        self.train_mrr_metrics = nn.ModuleDict()
        self.val_mrr_metrics = nn.ModuleDict()
        self.test_mrr_metrics = nn.ModuleDict()

        for k in self.ranking_ks:
            key = f"at_{k}"
            self.train_recall_metrics[key] = retrieval.RetrievalRecall(top_k=k)
            self.val_recall_metrics[key] = retrieval.RetrievalRecall(top_k=k)
            self.test_recall_metrics[key] = retrieval.RetrievalRecall(top_k=k)

            self.train_ndcg_metrics[key] = retrieval.RetrievalNormalizedDCG(top_k=k)
            self.val_ndcg_metrics[key] = retrieval.RetrievalNormalizedDCG(top_k=k)
            self.test_ndcg_metrics[key] = retrieval.RetrievalNormalizedDCG(top_k=k)

            self.train_mrr_metrics[key] = retrieval.RetrievalMRR(top_k=k)
            self.val_mrr_metrics[key] = retrieval.RetrievalMRR(top_k=k)
            self.test_mrr_metrics[key] = retrieval.RetrievalMRR(top_k=k)

    @classmethod
    def from_datamodule(
        cls,
        datamodule,
        embedding_dim: int = 64,
        user_hidden_dims: Sequence[int] = (128,),
        item_hidden_dims: Sequence[int] = (128,),
        normalize_embeddings: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        ranking_ks: Sequence[int] = (5, 10),
        scheduler_name: str = "cosine",
        scheduler_t_max: int = 10,
        scheduler_eta_min: float = 1e-6,
    ) -> "TwoTowerModel":
        return cls(
            config=datamodule.config,
            **datamodule.model_feature_config,
            embedding_dim=embedding_dim,
            user_hidden_dims=user_hidden_dims,
            item_hidden_dims=item_hidden_dims,
            normalize_embeddings=normalize_embeddings,
            lr=lr,
            weight_decay=weight_decay,
            ranking_ks=ranking_ks,
            scheduler_name=scheduler_name,
            scheduler_t_max=scheduler_t_max,
            scheduler_eta_min=scheduler_eta_min,
        )

    def _encode_cat_features(
        self,
        cat_tensor: torch.Tensor,
        feature_names: Sequence[str],
        embedding_layers: nn.ModuleDict,
    ) -> torch.Tensor:
        if len(feature_names) == 0:
            return torch.empty((cat_tensor.size(0), 0), device=cat_tensor.device)

        embeddings = [
            embedding_layers[col](cat_tensor[:, idx])
            for idx, col in enumerate(feature_names)
        ]
        return torch.cat(embeddings, dim=1)

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_embeddings:
            return x
        return F.normalize(x, p=2, dim=1)

    def encode_user(self, user_cat: torch.Tensor, user_num: torch.Tensor) -> torch.Tensor:
        user_cat_embed = self._encode_cat_features(
            cat_tensor=user_cat,
            feature_names=self.user_cat_feature_names,
            embedding_layers=self.user_cat_embeddings,
        )
        user_input = torch.cat([user_cat_embed, user_num], dim=1)
        user_embedding = self.user_tower(user_input)
        return self._maybe_normalize(user_embedding)

    def encode_item(self, item_cat: torch.Tensor, item_num: torch.Tensor) -> torch.Tensor:
        item_cat_embed = self._encode_cat_features(
            cat_tensor=item_cat,
            feature_names=self.item_cat_feature_names,
            embedding_layers=self.item_cat_embeddings,
        )
        item_input = torch.cat([item_cat_embed, item_num], dim=1)
        item_embedding = self.item_tower(item_input)
        return self._maybe_normalize(item_embedding)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        user_embedding = self.encode_user(
            user_cat=batch["user_cat"],
            user_num=batch["user_num"],
        )
        item_embedding = self.encode_item(
            item_cat=batch["item_cat"],
            item_num=batch["item_num"],
        )
        logit = (user_embedding * item_embedding).sum(dim=1)
        prob = torch.sigmoid(logit)
        return {
            "logit": logit,
            "prob": prob,
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
        }

    def score_candidates(
        self,
        user_embedding: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return user_embedding @ item_embeddings.T

    def recommend_topk(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        candidate_item_cat: torch.Tensor,
        candidate_item_num: torch.Tensor,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_embedding = self.encode_user(user_cat=user_cat, user_num=user_num)
        item_embeddings = self.encode_item(
            item_cat=candidate_item_cat,
            item_num=candidate_item_num,
        )
        scores = self.score_candidates(
            user_embedding=user_embedding,
            item_embeddings=item_embeddings,
        )
        safe_top_k = min(top_k, item_embeddings.size(0))
        top_scores, top_indices = torch.topk(scores, k=safe_top_k, dim=1)
        return top_indices, top_scores

    def recommend_item_ids(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        candidate_item_cat: torch.Tensor,
        candidate_item_num: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_indices, top_scores = self.recommend_topk(
            user_cat=user_cat,
            user_num=user_num,
            candidate_item_cat=candidate_item_cat,
            candidate_item_num=candidate_item_num,
            top_k=top_k,
        )
        top_item_ids = candidate_item_ids[top_indices]
        return top_item_ids, top_scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.scheduler_t_max,
                eta_min=self.hparams.scheduler_eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return {"optimizer": optimizer}

    def _get_metric_groups(self, step: str):
        if step == "train":
            return (
                self.train_loss_metric,
                self.train_recall_metrics,
                self.train_ndcg_metrics,
                self.train_mrr_metrics,
            )
        if step == "valid":
            return (
                self.val_loss_metric,
                self.val_recall_metrics,
                self.val_ndcg_metrics,
                self.val_mrr_metrics,
            )
        if step == "test":
            return (
                self.test_loss_metric,
                self.test_recall_metrics,
                self.test_ndcg_metrics,
                self.test_mrr_metrics,
            )
        raise ValueError(f"Unsupported step: {step}")

    def basic_step(self, batch: dict[str, torch.Tensor], step: str) -> dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        loss = self.loss_fn(outputs["logit"], batch["label"])

        loss_metric, recall_metrics, ndcg_metrics, mrr_metrics = self._get_metric_groups(step)
        loss_metric.update(loss.detach())

        metrics = {f"{step}/loss": loss}
        self.log(
            f"{step}/loss",
            loss_metric,
            prog_bar=(step != "test"),
            on_step=(step == "train"),
            on_epoch=True,
            batch_size=batch["label"].size(0),
        )

        if "query_id" in batch and step != "train":
            preds = outputs["logit"].detach()
            target = batch["label"].long()
            indexes = batch["query_id"]

            for k in self.ranking_ks:
                key = f"at_{k}"
                recall_metrics[key].update(preds, target, indexes=indexes)
                ndcg_metrics[key].update(preds, target, indexes=indexes)
                mrr_metrics[key].update(preds, target, indexes=indexes)

                self.log(
                    f"{step}/recall@{k}",
                    recall_metrics[key],
                    prog_bar=(step != "test"),
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch["label"].size(0),
                )
                self.log(
                    f"{step}/ndcg@{k}",
                    ndcg_metrics[key],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch["label"].size(0),
                )
                self.log(
                    f"{step}/mrr@{k}",
                    mrr_metrics[key],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch["label"].size(0),
                )

        return metrics

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self.basic_step(batch, "train")
        return metrics["train/loss"]

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self.basic_step(batch, "valid")
        return metrics["valid/loss"]

    @torch.no_grad()
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self.basic_step(batch, "test")
        return metrics["test/loss"]

    def on_train_epoch_start(self) -> None:
        logger.info(
            "Train epoch %s started. lr=%.6g",
            self.current_epoch,
            self.lr,
        )

    def on_train_epoch_end(self) -> None:
        train_loss = self.trainer.callback_metrics.get("train/loss")
        logger.info(
            "Train epoch %s finished. train/loss=%s",
            self.current_epoch,
            self._format_metric_value(train_loss),
        )

    def on_validation_epoch_start(self) -> None:
        logger.info("Validation epoch %s started.", self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        metric_parts = [
            f"val/loss={self._format_metric_value(self.trainer.callback_metrics.get('valid/loss'))}",
        ]
        for k in self.ranking_ks:
            metric_parts.append(
                f"valid/recall@{k}={self._format_metric_value(self.trainer.callback_metrics.get(f'valid/recall@{k}'))}"
            )
            metric_parts.append(
                f"valid/ndcg@{k}={self._format_metric_value(self.trainer.callback_metrics.get(f'valid/ndcg@{k}'))}"
            )
            metric_parts.append(
                f"valid/mrr@{k}={self._format_metric_value(self.trainer.callback_metrics.get(f'valid/mrr@{k}'))}"
            )
        logger.info(
            "Validation epoch %s finished. %s",
            self.current_epoch,
            ", ".join(metric_parts),
        )

    def on_test_epoch_start(self) -> None:
        logger.info("Test epoch started.")

    def on_test_epoch_end(self) -> None:
        metric_parts = [
            f"test/loss={self._format_metric_value(self.trainer.callback_metrics.get('test/loss'))}",
        ]
        for k in self.ranking_ks:
            metric_parts.append(
                f"test/recall@{k}={self._format_metric_value(self.trainer.callback_metrics.get(f'test/recall@{k}'))}"
            )
            metric_parts.append(
                f"test/ndcg@{k}={self._format_metric_value(self.trainer.callback_metrics.get(f'test/ndcg@{k}'))}"
            )
            metric_parts.append(
                f"test/mrr@{k}={self._format_metric_value(self.trainer.callback_metrics.get(f'test/mrr@{k}'))}"
            )
        logger.info("Test epoch finished. %s", ", ".join(metric_parts))

    @staticmethod
    def _format_metric_value(value: object) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, torch.Tensor):
            return f"{value.detach().float().item():.6f}"
        return f"{float(value):.6f}"
