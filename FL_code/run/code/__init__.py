"""
FedAgri — Federated Learning for Non-IID Maize Disease Detection.

Full implementation package (``run/code/``):
  models       — ResNet/MobileNet/EfficientNet + FedCAMNet (KAN+SE) + MD2/MiniNetPP
  aggregation  — FedAvg family + FedCADR (fed_car / fed_car_plus) + FedAgri + FedAdam …
  dataset      — MaizeDNIID Non-IID partitioning
  training     — local client train/eval
  federated_loop — sequential / threaded / sync FL orchestration
"""

from .config import FLConfig, DEFAULT_FL_CONFIG, NUM_CLASSES, VAL_DIR, TRAIN_DIR, FEDERATED_ROOT
from .dataset import (
    create_federated_dataset,
    FederatedDataset,
    load_all_client_datasets,
    count_images_per_class,
    visualize_data_distribution,
    federated_output_path,
    build_global_class_mapping,
    get_fl_transform,
)
from .models import (
    SEBlock,
    ChannelAttention,
    FeatureContext,
    ContextAwareModule,
    DualPooling,
    KANLayer,
    KANLayer_111,
    KANLayerRBF,
    ContextKANHead,
    FederatedMobileNetKAN,
    FedCAMNet,
    FederatedMobileNetKAN_Ablation,
    ContextAwareMobileNet,
    FinalFLModel,
    MD2,
    MiniNetPP,
    SimpleCNN,
    get_resnet18_model,
    get_mobilenet_model,
    get_efficientnet_model,
    get_squeezenet_model,
    get_shufflenet_model,
    get_densenet_model,
    get_cnn_model,
    CNNModel,
    get_model,
)
from .aggregation import (
    federated_avg,
    weighted_federated_avg,
    federated_sgd,
    fedprox,
    differential_privacy_aggregation,
    krum,
    median_aggregation,
    trimmed_mean,
    fednova,
    federated_momentum,
    divergence_aware_agg,
    fedadam,
    fedadagrad,
    FedAgri,
    global_alignment_loss,
    optimize_global_model,
    fed_car,
    fed_car_plus,
    FedCADR,
    fedcadr,
    fed_dca,
    fedavg_inplace,
    aggregate,
    get_aggregator,
    AGGREGATOR_REGISTRY,
)
from .federated_loop import (
    run_federated_learning_threaded,
    run_federated_learning_sequential,
    run_federated_learning_homogeneous,
    run_federated_train_sync,
    StartFedLearnining,
)
from .metrics import (
    evaluate,
    evaluate_client,
    flatten_state_dict,
    compare_to_average,
    histories_to_df,
)
from .run_fed import run_fed, load_run_fed_clients, local_train_run_fed
from .training import evaluate_model, plot_client_performance, split_dataset, train_client_sync

__all__ = [
    "FLConfig", "DEFAULT_FL_CONFIG", "NUM_CLASSES", "TRAIN_DIR", "VAL_DIR", "FEDERATED_ROOT",
    "create_federated_dataset", "FederatedDataset", "load_all_client_datasets",
    "count_images_per_class", "visualize_data_distribution", "federated_output_path",
    "build_global_class_mapping", "get_fl_transform",
    "SEBlock", "ChannelAttention", "FeatureContext", "ContextAwareModule", "DualPooling",
    "KANLayer", "KANLayer_111", "KANLayerRBF", "ContextKANHead",
    "FederatedMobileNetKAN", "FedCAMNet", "FederatedMobileNetKAN_Ablation",
    "ContextAwareMobileNet", "FinalFLModel", "MD2", "MiniNetPP", "SimpleCNN",
    "get_resnet18_model", "get_mobilenet_model", "get_efficientnet_model",
    "get_squeezenet_model", "get_shufflenet_model", "get_densenet_model",
    "get_cnn_model", "CNNModel", "get_model",
    "federated_avg", "weighted_federated_avg", "federated_sgd", "fedprox",
    "differential_privacy_aggregation", "krum", "median_aggregation", "trimmed_mean",
    "fednova", "federated_momentum", "divergence_aware_agg", "fedadam", "fedadagrad",
    "FedAgri", "global_alignment_loss", "optimize_global_model",
    "fed_car", "fed_car_plus", "FedCADR", "fedcadr", "fed_dca", "fedavg_inplace",
    "aggregate", "get_aggregator", "AGGREGATOR_REGISTRY",
    "run_federated_learning_threaded", "run_federated_learning_sequential",
    "run_federated_learning_homogeneous", "run_federated_train_sync", "StartFedLearnining",
    "evaluate_model", "plot_client_performance", "split_dataset", "train_client_sync",
    "evaluate", "evaluate_client", "flatten_state_dict", "compare_to_average",
    "histories_to_df", "run_fed", "load_run_fed_clients", "local_train_run_fed",
]
