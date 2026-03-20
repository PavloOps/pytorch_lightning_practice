from dataclasses import dataclass, field

from torch.cuda import is_available


@dataclass
class GeneralConfig:
    seed: int = 2025


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    accelerator: str = "cuda" if is_available() else "cpu"
    devices: int = 1
    precision: str = "16-mixed" if is_available() else "32-true"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0
    detect_anomaly: bool = False
    log_every_n_steps: int = 20
    check_val_every_n_epoch: int = 1


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-5
    embedding_dim: int = 64
    negative_ratio: int = 10
    eval_negative_ratio: int = 20
    hard_negative_popular_share: float = 0.4
    hard_negative_same_category_share: float = 0.4


@dataclass
class DataConfig:
    data_dir: str = "final_project/data"
    train_file_name: str = "train_dataset_prepared.csv"
    val_file_name: str | None = "valid_dataset_prepared.csv"
    test_file_name: str | None = None
    batch_size: int = 1024
    num_workers: int = 4
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    label_col: str = "label"
    add_month_feature: bool = True
    drop_last: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    user_cat_cols: tuple[str, ...] = ("user_id", "region", "city")
    item_cat_cols: tuple[str, ...] = (
        "item_id",
        "brand",
        "category",
        "country",
        "inn",
        "owner",
    )
    user_num_cols: tuple[str, ...] = ("user_total_cnt", "user_unique_items")
    item_num_cols: tuple[str, ...] = ("avg_price",)


@dataclass
class InferenceConfig:
    top_k: int = 10


@dataclass
class MetricsConfig:
    ranking_ks: tuple[int, ...] = (5, 10)


@dataclass
class OptimizationConfig:
    scheduler_name: str = "cosine"  # one of: none, cosine, onecycle
    scheduler_t_max: int = 10
    scheduler_eta_min: float = 1e-6
    onecycle_pct_start: float = 0.1
    use_torch_compile: bool = False


@dataclass
class CallbacksConfig:
    early_stopping_patience: int = 3
    save_top_k: int = 3
    use_rich_progress_bar: bool = True
    use_rich_model_summary: bool = True
    use_swa: bool = False
    swa_lrs: float = 1e-3
    swa_epoch_start: float = 0.8


@dataclass
class ClearMLConfig:
    enabled: bool = True
    project_name: str = "RecSys / Two Tower"
    task_name: str = "two-tower-training"
    tags: list[str] = field(default_factory=lambda: ["recsys", "two-tower", "training"])
    reuse_last_task_id: bool = False
    continue_last_task: bool = False
    output_uri: str | bool | None = None
    prompt_missing_credentials: bool = True
    connect_full_config: bool = True
    log_dataset_preview: bool = True
    dataset_preview_rows: int = 20
    upload_best_checkpoint: bool = True


@dataclass
class DebugEvalConfig:
    enabled: bool = True
    raw_train_path: str | None = None
    per_case: int = 25
    candidates_per_sample: int = 100
    seed: int = 42
    preview_rows: int = 20
    report_top_cases: int = 20


@dataclass
class CFG:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)
    debug_eval: DebugEvalConfig = field(default_factory=DebugEvalConfig)
