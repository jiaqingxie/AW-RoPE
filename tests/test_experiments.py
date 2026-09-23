import json
from collections import Counter

import pytest
import torch
from torch_geometric.data import Batch, Data

from aw_rope.experiments import train as train_module
from aw_rope.experiments.datasets import DatasetBundle, _stratified_search_split
from aw_rope.experiments.grid import synthetic_grid
from aw_rope.experiments.models import GraphPredictionModel
from aw_rope.experiments.train import TrialConfig


def tiny_batch() -> Batch:
    graphs = []
    for offset in (0.0, 1.0):
        graphs.append(
            Data(
                x=torch.tensor([[offset], [1.0 + offset], [2.0 + offset]]),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                y=torch.tensor([offset]),
            )
        )
    return Batch.from_data_list(graphs)


def test_stratified_search_split_is_exact_balanced_and_deterministic() -> None:
    dataset = [
        Data(y=torch.tensor([label]))
        for label, count in enumerate((7, 11, 17))
        for _ in range(count)
    ]
    train, validation = _stratified_search_split(dataset, validation_size=9)
    train_again, validation_again = _stratified_search_split(
        dataset, validation_size=9
    )
    assert train.indices == train_again.indices
    assert validation.indices == validation_again.indices
    assert len(train) == 26
    assert len(validation) == 9
    assert Counter(int(validation[index].y.item()) for index in range(len(validation))) == {
        0: 2,
        1: 3,
        2: 4,
    }


@pytest.mark.parametrize("method", ["nope", "aw", "nb", "multiscale-nb"])
def test_graph_prediction_methods_train(method: str) -> None:
    batch = tiny_batch()
    model = GraphPredictionModel(
        1,
        1,
        hidden_dim=8,
        num_layers=2,
        method=method,
        num_steps=2,
    )
    output = model(batch)
    output.square().mean().backward()
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_synthetic_grid_is_exhaustive_and_unique() -> None:
    trials = synthetic_grid()
    assert len(trials) == 110
    assert len({trial.trial_id for trial in trials}) == len(trials)
    assert {trial.dataset for trial in trials} == {
        "monochromatic-0",
        "monochromatic-5",
        "monochromatic-10",
        "monochromatic-15",
        "watts-strogatz-spd",
    }


@pytest.mark.parametrize(
    "variant",
    ["rezero", "rezero-normalized", "single-normalized", "structural-rms"],
)
def test_gin_aw_repair_variants_are_finite(variant: str) -> None:
    batch = tiny_batch()
    model = GraphPredictionModel(
        1,
        3,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        method="aw",
        num_steps=2,
        z=0.6,
        gin_aw_variant=variant,
    )
    output = model(batch)
    output.square().mean().backward()
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert all(float(branch.gate.detach()) == 0.0 for branch in model.analytic_branches)
    assert all(
        torch.isfinite(branch.last_displacement_rms)
        for branch in model.analytic_branches
    )


@pytest.mark.parametrize(
    "variant",
    ["rezero", "rezero-normalized", "single-normalized", "structural-rms"],
)
def test_gin_aw_rezero_starts_from_paired_nope_function(variant: str) -> None:
    batch = tiny_batch()
    common = dict(
        input_dim=1,
        output_dim=3,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        num_steps=2,
        z=0.6,
    )
    torch.manual_seed(17)
    nope = GraphPredictionModel(method="nope", **common).eval()
    torch.manual_seed(17)
    aw = GraphPredictionModel(method="aw", gin_aw_variant=variant, **common).eval()
    assert torch.equal(nope(batch), aw(batch))


def test_gin_structural_rms_handles_promoted_degree_statistics() -> None:
    batch = tiny_batch()
    model = GraphPredictionModel(
        1,
        3,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        method="aw",
        num_steps=2,
        z=0.6,
        gin_aw_variant="structural-rms",
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(batch)
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_structural_rms_normalization_preserves_bfloat16_dtype() -> None:
    from aw_rope.experiments.models import _AnalyticBranch

    branch = _AnalyticBranch(
        8,
        "aw",
        num_steps=2,
        z=0.6,
        learnable_frequencies=False,
        position_dim=0,
        zero_init_gate=True,
        normalize_resolvent=True,
        normalize_displacement_rms=True,
    )
    displacement = torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.bfloat16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    node_batch = torch.tensor([0, 0, 1, 1])

    normalized = branch._normalize_displacement(
        displacement, edge_index, node_batch
    )

    assert normalized.dtype == torch.bfloat16
    assert torch.isfinite(normalized).all()


def test_gin_diagnostics_counts_sequence_predictions(monkeypatch) -> None:
    """Code2 logits have a sequence axis and must still yield class counts."""
    batch = tiny_batch()
    model = GraphPredictionModel(
        1,
        4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        method="nope",
    )
    sequence_logits = torch.tensor(
        [
            [[3.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, -1.0]],
            [[0.0, 1.0, 4.0, -1.0], [0.0, 1.0, 2.0, 5.0]],
        ]
    )
    monkeypatch.setattr(model, "forward", lambda _batch: sequence_logits)

    diagnostics = train_module._gin_model_diagnostics(
        model,
        [batch],
        torch.device("cpu"),
    )

    assert diagnostics["predicted_class_counts"] == [1, 1, 1, 1]


def test_scheduler_t_max_can_preserve_screening_trajectory(monkeypatch) -> None:
    config = TrialConfig(dataset="ogbg-molhiv", method="aw", epochs=250)
    monkeypatch.delenv("AW_ROPE_SCHEDULER_T_MAX", raising=False)
    assert train_module._scheduler_t_max(config) == 250
    monkeypatch.setenv("AW_ROPE_SCHEDULER_T_MAX", "120")
    assert train_module._scheduler_t_max(config) == 120


def test_optimizer_step_schedule_drops_incomplete_effective_batch() -> None:
    config = TrialConfig(
        dataset="shapenet",
        method="aw",
        batch_size=2,
        gradient_accumulation_steps=512,
        scheduler_step_unit="optimizer-step",
    )
    steps, usable = train_module._optimizer_schedule_shape(config, 6068)
    assert steps == 11
    assert usable == 5632


def test_epoch_schedule_preserves_partial_accumulation_window() -> None:
    config = TrialConfig(
        dataset="tiny",
        method="aw",
        gradient_accumulation_steps=3,
        scheduler_step_unit="epoch",
    )
    steps, usable = train_module._optimizer_schedule_shape(config, 8)
    assert steps == 3
    assert usable == 8


def test_per_epoch_validation_and_test_are_persisted_and_test_selects_checkpoint(
    tmp_path, monkeypatch
) -> None:
    def graph(label: int, offset: float) -> Data:
        return Data(
            x=torch.tensor([[offset], [offset + 0.5], [offset + 1.0]]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            y=torch.tensor([label]),
        )

    bundle = DatasetBundle(
        name="tiny",
        task="graph-classification",
        train=[graph(0, 0.0), graph(1, 1.0), graph(0, 0.2), graph(1, 1.2)],
        validation=[graph(0, 0.1), graph(1, 1.1)],
        test=[graph(0, 0.3), graph(1, 1.3)],
        input_dim=1,
        output_dim=2,
        metric="accuracy",
        metric_mode="max",
    )
    monkeypatch.setattr(train_module, "load_dataset", lambda *_args, **_kwargs: bundle)
    config = TrialConfig(
        dataset="tiny",
        method="nope",
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        num_steps=2,
        learning_rate=1e-2,
        batch_size=2,
        epochs=2,
        patience=2,
        num_workers=0,
        evaluate_test_each_epoch=True,
        selection_split="test",
    )

    result = train_module.run_trial(config, tmp_path / "data", tmp_path / "runs")
    trial_dir = tmp_path / "runs" / "tiny" / "nope" / config.trial_id
    rows = [
        json.loads(line)
        for line in (trial_dir / "epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 2
    assert all("accuracy" in row["validation"] for row in rows)
    assert all("accuracy" in row["test"] for row in rows)
    assert all(row["selection_split"] == "test" for row in rows)
    assert result["selection_split"] == "test"
    assert result["test_selection_bias"] is True
    checkpoint = torch.load(trial_dir / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["selection_split"] == "test"
    assert checkpoint["best_epoch"] == result["best_epoch"]


def test_single_trial_resumes_from_atomic_last_checkpoint(
    tmp_path, monkeypatch
) -> None:
    def graph(label: int, offset: float) -> Data:
        return Data(
            x=torch.tensor([[offset], [offset + 0.5], [offset + 1.0]]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            y=torch.tensor([label]),
        )

    bundle = DatasetBundle(
        name="tiny-resume",
        task="graph-classification",
        train=[graph(0, 0.0), graph(1, 1.0), graph(0, 0.2), graph(1, 1.2)],
        validation=[graph(0, 0.1), graph(1, 1.1)],
        test=[graph(0, 0.3), graph(1, 1.3)],
        input_dim=1,
        output_dim=2,
        metric="accuracy",
        metric_mode="max",
    )
    monkeypatch.setattr(train_module, "load_dataset", lambda *_args, **_kwargs: bundle)
    config = TrialConfig(
        dataset="tiny-resume",
        method="nope",
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-2,
        batch_size=2,
        epochs=2,
        patience=2,
        num_workers=0,
        evaluate_test_each_epoch=True,
    )
    original_save = train_module._atomic_torch_save
    interrupted = False

    def save_then_interrupt(payload, path):
        nonlocal interrupted
        original_save(payload, path)
        if path.name == "last.pt" and payload["epoch"] == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated preemption")

    monkeypatch.setattr(train_module, "_atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        train_module.run_trial(config, tmp_path / "data", tmp_path / "runs")

    trial_dir = tmp_path / "runs" / config.dataset / config.method / config.trial_id
    last = torch.load(trial_dir / "last.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 1

    monkeypatch.setattr(train_module, "_atomic_torch_save", original_save)
    result = train_module.run_trial(config, tmp_path / "data", tmp_path / "runs")
    rows = [
        json.loads(line)
        for line in (trial_dir / "epoch_metrics.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert len(result["history"]) == 2
    assert torch.load(
        trial_dir / "last.pt", map_location="cpu", weights_only=False
    )["epoch"] == 2
