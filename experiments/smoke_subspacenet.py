"""Smoke test for the upstream SubspaceNet submodule.

Exercises create_dataset -> train (1 epoch) -> evaluate with tiny sizes,
purely to confirm the environment and submodule wire up end-to-end.
Not a real training run.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

SUBSPACENET_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "subspacenet"
sys.path.insert(0, str(SUBSPACENET_ROOT))
os.chdir(SUBSPACENET_ROOT)

import torch

from src.system_model import SystemModelParams
from src.models import ModelGenerator
from src.data_handler import create_dataset
from src.training import TrainingParams, train, simulation_summary, get_simulation_filename
from src.criterions import set_criterions
from src.evaluation import evaluate
from src.plotting import initialize_figures
from src.utils import set_unified_seed

set_unified_seed()

samples_size = 256
train_test_ratio = 0.1
tau = 8

system_model_params = (
    SystemModelParams()
    .set_parameter("N", 8)
    .set_parameter("M", 3)
    .set_parameter("T", 200)
    .set_parameter("snr", 10)
    .set_parameter("signal_type", "NarrowBand")
    .set_parameter("signal_nature", "non-coherent")
    .set_parameter("eta", 0)
    .set_parameter("bias", 0.05)
    .set_parameter("sv_noise_var", 0)
)

model_config = (
    ModelGenerator()
    .set_model_type("SubspaceNet")
    .set_diff_method("esprit")
    .set_tau(tau)
    .set_model(system_model_params)
)

datasets_path = SUBSPACENET_ROOT / "data" / "datasets" / "uniform_bias_spacing"
(datasets_path / "train").mkdir(parents=True, exist_ok=True)
(datasets_path / "test").mkdir(parents=True, exist_ok=True)
saving_path = SUBSPACENET_ROOT / "data" / "weights"
saving_path.mkdir(parents=True, exist_ok=True)

print("=== creating tiny train set ===")
train_dataset, _, _ = create_dataset(
    system_model_params=system_model_params,
    samples_size=samples_size,
    model_type=model_config.model_type,
    tau=model_config.tau,
    save_datasets=False,
    datasets_path=datasets_path,
    true_doa=None,
    phase="train",
)

print("=== creating tiny test set ===")
test_dataset, generic_test_dataset, samples_model = create_dataset(
    system_model_params=system_model_params,
    samples_size=max(8, int(train_test_ratio * samples_size)),
    model_type=model_config.model_type,
    tau=model_config.tau,
    save_datasets=False,
    datasets_path=datasets_path,
    true_doa=None,
    phase="test",
)

print("=== training 1 epoch ===")
simulation_parameters = (
    TrainingParams()
    .set_batch_size(32)
    .set_epochs(1)
    .set_model(model=model_config)
    .set_optimizer(optimizer="Adam", learning_rate=1e-4, weight_decay=1e-9)
    .set_training_dataset(train_dataset)
    .set_schedular(step_size=1, gamma=0.5)
    .set_criterion()
)

simulation_filename = get_simulation_filename(
    system_model_params=system_model_params, model_config=model_config
)
simulation_summary(
    system_model_params=system_model_params,
    model_type=model_config.model_type,
    parameters=simulation_parameters,
    phase="training",
)

model, _, _ = train(
    training_parameters=simulation_parameters,
    model_name=simulation_filename,
    saving_path=saving_path,
)

print("=== evaluating ===")
model_test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, drop_last=False
)
generic_test_loader = torch.utils.data.DataLoader(
    generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
)
criterion, subspace_criterion = set_criterions("rmse")
figures = initialize_figures()

evaluate(
    model=model,
    model_type=model_config.model_type,
    model_test_dataset=model_test_loader,
    generic_test_dataset=generic_test_loader,
    criterion=criterion,
    subspace_criterion=subspace_criterion,
    system_model=samples_model,
    figures=figures,
    plot_spec=False,
)

print("=== smoke test OK ===")
