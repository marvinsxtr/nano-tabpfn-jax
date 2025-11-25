import random
import time
from collections.abc import Callable, Generator

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import torch
from jaxtyping import Array, Float
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import NanoTabPFNClassifier, NanoTabPFNModel


def set_randomness_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


set_randomness_seed(0)


def get_default_device() -> str:
    """Get the default device for PyTorch computations."""
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


# Prepare datasets
datasets = []
datasets.append(train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0))


def eval(classifier: NanoTabPFNClassifier) -> dict[str, float]:  # noqa: A001
    """Evaluate the classifier on multiple datasets and return average scores."""
    scores = {"roc_auc": 0, "acc": 0, "balanced_acc": 0}
    for X_train, X_test, y_train, y_test in datasets:
        classifier.fit(X_train, y_train)
        prob = classifier.predict_proba(X_test)
        pred = prob.argmax(axis=1)  # avoid a second forward pass by not calling predict
        if prob.shape[1] == 2:
            prob = prob[:, :1]
        scores["roc_auc"] += float(roc_auc_score(y_test, prob, multi_class="ovr"))
        scores["acc"] += float(accuracy_score(y_test, pred))
        scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    scores = {k: v / len(datasets) for k, v in scores.items()}
    return scores


@eqx.filter_jit
def compute_loss(
    model: NanoTabPFNModel,
    data: tuple[Float[Array, "batch_size num_rows num_features"], Float[Array, "batch_size num_rows"]],
    targets: Float[Array, "batch_size num_rows"],
    train_mask: Float[Array, "batch_size num_rows"],
) -> tuple[Float[Array, ""], dict]:
    """Compute cross-entropy loss for the model.

    Args:
        model: The NanoTabPFNModel
        data: Tuple of (x, y) - x has all rows, y has only training labels
        targets: Ground truth labels for all rows
        train_mask: boolean mask indicating training rows (batch_size, num_rows)

    Returns:
        Tuple of (loss, aux_dict)
    """
    output = jax.vmap(model)(data[0], data[1], train_mask)  # (batch_size, num_rows, num_outputs)

    test_mask = ~train_mask  # (batch_size, num_rows)

    # Reshape for cross-entropy: (batch_size * num_rows, num_classes)
    output_flat = output.reshape(-1, output.shape[-1])
    targets_flat = targets.reshape(-1).astype(jnp.int32)
    mask_flat = test_mask.reshape(-1)

    # Compute loss only on test examples
    loss_per_sample = optax.softmax_cross_entropy_with_integer_labels(output_flat, targets_flat, where=mask_flat)

    # Average only over test samples
    loss = loss_per_sample.sum() / mask_flat.sum()

    aux = {"loss": loss}
    return loss, aux


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    data: tuple[Float[Array, "batch_size num_rows num_features"], Float[Array, "batch_size num_train"]],
    targets: Float[Array, "batch_size num_rows"],
    train_mask: Float[Array, "batch_size num_rows"],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[Array, dict, eqx.Module, optax.OptState]:
    """Perform one optimization step using computed gradients.

    JIT-compiled function that computes loss and gradients, applies optimizer
    updates, and returns updated model state.

    Args:
        model: Current model parameters.
        data: Training data tuple (x, y_train).
        targets: Target labels for all rows.
        train_mask: boolean mask indicating training rows (batch_size, num_rows)
        opt_state: Current optimizer state.
        optimizer: Optax optimizer transformation.

    Returns:
        Tuple of (loss, aux, updated_model, updated_opt_state).
    """
    model = eqx.nn.inference_mode(model, value=False)
    loss_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = loss_fn(model, data, targets, train_mask)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return loss, aux, model, opt_state


def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    lr: float = 1e-4,
    steps_per_eval: int = 10,
    eval_func: Callable | None = None,
) -> tuple[NanoTabPFNModel, list[tuple[float, dict]]]:
    """Trains our model on the given prior using cross-entropy loss.

    Args:
        model: NanoTabPFNModel in JAX/Equinox
        prior: DataLoader providing training batches
        lr: learning rate
        steps_per_eval: how many steps we wait before running evaluation again
        eval_func: a function that takes in a classifier and returns a dict containing the average scores
                   for some metrics and datasets

    Returns:
        model: trained JAX model
        eval_history: list containing eval history, each entry is the real time used for training so far together
                      with a dict mapping metric names to their average values across a list of datasets
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.contrib.schedule_free_adamw(learning_rate=lr, weight_decay=0.0)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_time = 0
    eval_history = []

    try:
        for step, full_data in enumerate(prior):
            if step == 0:
                print(f"Step {step}: Starting first training step (compiling)...")
            step_start_time = time.time()

            x, y = full_data["x"], full_data["y"]
            train_test_split_index = full_data["train_test_split_index"]

            # Pad to 10 features if needed
            if x.shape[2] < 10:
                padding = np.zeros((x.shape[0], x.shape[1], 10 - x.shape[2]))
                x_padded = np.concatenate([x, padding], axis=2)
            else:
                x_padded = x

            train_mask = np.arange(x.shape[1]) < train_test_split_index  # x.shape[1] is num_rows
            train_mask = np.broadcast_to(train_mask, (x.shape[0], x.shape[1]))  # (batch_size, num_rows)

            data = (x_padded, y)

            loss, _, model, opt_state = make_step(model, data, y, train_mask, opt_state, optimizer)

            total_loss = float(loss)
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            if step == 0:
                print(f"Step {step} completed in {step_train_duration:.1f}s (compilation + execution)")

            # Print progress every step
            if step > 0 and step % 10 == 0:
                print(f"Step {step}: loss {total_loss:7.4f}, time {step_train_duration:.2f}s")

            # Evaluate
            if step % steps_per_eval == steps_per_eval - 1 and eval_func is not None:
                classifier = NanoTabPFNClassifier(model)
                scores = eval_func(classifier)
                eval_history.append((train_time, scores))
                score_str = " | ".join([f"{k} {v:7.4f}" for k, v in scores.items()])
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")
            elif step % steps_per_eval == steps_per_eval - 1 and eval_func is None:
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f}")

    except KeyboardInterrupt:
        pass

    return model, eval_history


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename: str, num_steps: int, batch_size: int, device: torch.device | None = None) -> None:  # noqa: ARG002
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.pointer = 0
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self) -> Generator[dict[str, np.ndarray]]:
        """Yield batches of data from the HDF5 file."""
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer : end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                x = f["X"][self.pointer : end, :max_seq_in_batch, :num_features]
                y = f["y"][self.pointer : end, :max_seq_in_batch]
                train_test_split_index = f["single_eval_pos"][self.pointer : end][0].item()

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield {
                    "x": x,
                    "y": y,
                    "train_test_split_index": train_test_split_index,
                }

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_steps
