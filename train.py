import random
import time
from collections.abc import Callable

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model import NanoTabPFNClassifier, NanoTabPFNModel


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_randomness_seed(0)

# Prepare datasets
datasets = []
datasets.append(train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0))


def eval(classifier):
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
    data: tuple[Float[Array, "batch_size num_rows num_features"], Float[Array, "batch_size num_train"]],
    targets: Float[Array, "batch_size num_test"],
    train_test_split_index: int,
) -> tuple[Float[Array, ""], dict]:
    """Compute cross-entropy loss for the model.

    Args:
        model: The NanoTabPFNModel
        data: Tuple of (x, y_train)
        targets: Ground truth labels for test data
        train_test_split_index: Number of training datapoints

    Returns:
        Tuple of (loss, aux_dict)
    """
    output = model(data, train_test_split_index=train_test_split_index)

    # Reshape for cross-entropy: (batch_size * num_test, num_classes)
    output = output.reshape(-1, output.shape[-1])
    targets = targets[:, train_test_split_index:].reshape(-1).astype(jnp.int32)

    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(output, targets).mean()

    aux = {"loss": loss}
    return loss, aux


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    data: tuple[Float[Array, "batch_size num_rows num_features"], Float[Array, "batch_size num_train"]],
    targets: Float[Array, "batch_size num_test"],
    train_test_split_index: int,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[Array, dict, eqx.Module, optax.OptState]:
    """Perform one optimization step using computed gradients.

    JIT-compiled function that computes loss and gradients, applies optimizer
    updates, and returns updated model state.

    Args:
        model: Current model parameters.
        data: Training data tuple (x, y_train).
        targets: Target labels.
        train_test_split_index: Number of training datapoints.
        opt_state: Current optimizer state.
        optimizer: Optax optimizer transformation.

    Returns:
        Tuple of (loss, aux, updated_model, updated_opt_state).
    """
    loss_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = loss_fn(model, data, targets, train_test_split_index)

    # Clip gradients
    grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, aux, model, opt_state


def train(
    model: NanoTabPFNModel,
    prior,
    key: PRNGKeyArray,
    lr: float = 1e-4,
    steps_per_eval: int = 10,
    eval_func: Callable = None,
):
    """Trains our model on the given prior using cross-entropy loss.

    Args:
        model: NanoTabPFNModel in JAX/Equinox
        prior: DataLoader providing training batches
        key: JAX random key
        lr: learning rate
        steps_per_eval: how many steps we wait before running evaluation again
        eval_func: a function that takes in a classifier and returns a dict containing the average scores
                   for some metrics and datasets

    Returns:
        model: trained JAX model
        eval_history: list containing eval history, each entry is the real time used for training so far together
                      with a dict mapping metric names to their average values across a list of datasets
    """
    optimizer = optax.adamw(learning_rate=lr, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_time = 0
    eval_history = []

    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]

            # Convert to JAX arrays
            x = jnp.array(full_data["x"])
            y = jnp.array(full_data["y"])

            data = (x, y[:, :train_test_split_index])
            targets = y

            loss, aux, model, opt_state = make_step(model, data, targets, train_test_split_index, opt_state, optimizer)

            total_loss = float(loss)
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

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


class PriorDumpDataLoader:
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
    """

    def __init__(self, filename, num_steps, batch_size):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.pointer = 0
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer : end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                x = f["X"][self.pointer : end, :max_seq_in_batch, :num_features]
                y = f["y"][self.pointer : end, :max_seq_in_batch]
                train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield {
                    "x": x,
                    "y": y,
                    "train_test_split_index": train_test_split_index[0].item(),
                }

    def __len__(self):
        return self.num_steps


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    model = NanoTabPFNModel(
        embedding_size=96, num_attention_heads=4, mlp_hidden_size=192, num_layers=3, num_outputs=2, key=key
    )
    prior = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=2500, batch_size=32)
    key = jr.PRNGKey(1)
    model, history = train(model, prior, key, lr=4e-3, steps_per_eval=25)
    print("Final evaluation:")
    print(eval(NanoTabPFNClassifier(model)))
