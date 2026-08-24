<h1 align="center">INF265 — Project 1: Backpropagation and Gradient Descent</h1>
<h3 align="center">Jakob Berg &amp; Tobias Munch</h3>

Course project for **INF265 – Deep Learning** (University of Bergen). The project has two
parts: implementing backpropagation from scratch for a small MLP, and training/comparing
MLPs with gradient descent on a binary CIFAR-10 task (**plane vs. bird**).

---

## Contents

| File | Description |
|------|-------------|
| [backpropagation.ipynb](backpropagation.ipynb) | Part 1: vectorized `backpropagation(model, y_true, y_pred)` for the `MyNet` class, verified against the provided tests |
| [tests_backpropagation.py](tests_backpropagation.py) | Provided test harness (toy dataset + MNIST subset) |
| [gradient_descent.ipynb](gradient_descent.ipynb) | Part 2: CIFAR-2 data pipeline, `MyNet` MLP, `train` / `train_manual_update`, grid search, evaluation |
| [Report.md](Report.md) / [Report.pdf](Report.pdf) | Full written report with derivations, figures and discussion |
| [imgs/](imgs/) | Figures used in the report |
| [gradient_descent_output.txt](gradient_descent_output.txt) | Reference training output |
| [Project 1 Description.pdf](Project%201%20Description.pdf), [project_checklist.pdf](project_checklist.pdf) | Task description and checklist |

---

## Part 1 — Backpropagation from scratch

`MyNet` is a fully connected network with `tanh` activations that stores its pre-activations
`z[l]` and activations `a[l]` during the forward pass. The task is to fill in the backward
pass manually, without `autograd`.

The implementation walks backwards from layer `L` to layer `1`:

```python
def backpropagation(model, y_true, y_pred):
    L = model.L
    delta = None
    for i in range(L, 0, -1):
        if i == L:                                  # output layer
            delta = 2 * (y_pred - y_true) * model.df[i](model.z[i])
        else:                                       # hidden layers
            W_next = model.fc[str(i+1)].weight
            delta = (delta @ W_next) * model.df[i](model.z[i])

        A_prev = model.a[i-1]
        model.dL_dw[i] = delta.t() @ A_prev          # ∂L/∂W[l]
        model.dL_db[i] = delta.sum(dim=0)            # ∂L/∂b[l]
```

- **Output layer:** `2 * (y_pred - y_true)` is the MSE derivative, multiplied by `f'(z[L])`.
- **Hidden layers:** `delta @ W_next` propagates the downstream error signal.
- **Weights:** an outer product `deltaᵀ @ a[l-1]`, so no nested Python loops are needed.
- **Biases:** the sum of `delta` over the batch.

Verified with `main_test(backpropagation, model, ...)` on both the toy network
`MyNet([2, 3, 2])` and an MNIST-shaped network `MyNet([24*24, 16, 10])`.

## Part 2 — Gradient descent on CIFAR-2

**Data.** CIFAR-10 restricted to *plane* (label 0) and *bird* (label 1), split 90/10 into
train/validation: **8980 / 1020 / 2000** images (train / val / test). Images are normalized
with mean and std computed from the **training set only**. The class distribution is nearly
balanced (50.2 % planes, 49.8 % birds).

**Model.** A configurable MLP (`nn.Sequential`) taking a flattened `3×32×32 = 3072` vector,
repeating `Linear → ReLU` blocks with optional dropout, and ending in `Linear(→ 2)` with no
activation (softmax is inside `CrossEntropyLoss`). Default hidden sizes: `[512, 128, 32]`.

**Training.** Two functions:
- `train(...)` — uses PyTorch's `optim.SGD`.
- `train_manual_update(...)` — updates every parameter by hand, with optional momentum and
  weight decay.

A `comparison()` helper trains two identical model copies with both functions and checks
that the per-epoch training losses agree to within `1e-8`, for three settings: no
regularization, weight decay only, and weight decay + momentum. All three match.

**Model selection.** `evaluate_model()` grid-searches four architectures
(`baseline`, `depth`, `width`, `dropout`) against SGD hyperparameters
(`lr = 0.01`, `weight_decay ∈ {0, 1e-4, 0.01}`, `momentum ∈ {0, 0.9}`), 20 epochs, batch
size 64, fixed seed. Models are ranked by **maximum validation accuracy**.

Best configuration:

```
Architecture:  width  (hidden sizes 1024, 128, 32)
lr = 0.01, weight_decay = 0.0, momentum = 0.0, dropout = 0.0
Best epoch: 18   Best validation accuracy: 0.863
```

**Results.** Test accuracy **0.863** (1724 / 2000 correct); errors are balanced across
classes (128 planes → bird, 148 birds → plane). Training loss decreases monotonically for
all architectures while validation loss eventually rises — clear overfitting. Added depth
did not help; the wider model generalized best. Misclassified images are largely ambiguous
cases (low resolution, motion blur, cluttered backgrounds), suggesting the main limitation
is the MLP architecture itself, which ignores spatial structure.

<p align="center">
  <img src="imgs/val_loss_best_by_acc.png" width="480"><br>
  <em>Training and validation loss for the best run of each architecture</em>
</p>

<p align="center">
  <img src="imgs/confusion_matrix.png" width="330">
  <img src="imgs/misclassifications.png" width="330"><br>
  <em>Test-set confusion matrix and example misclassifications</em>
</p>

See [Report.md](Report.md) for the full analysis.

---

## Running the project

Requirements: `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`.

```bash
pip install torch torchvision numpy matplotlib scikit-learn
jupyter notebook backpropagation.ipynb   # Part 1
jupyter notebook gradient_descent.ipynb  # Part 2
```

CIFAR-10 is downloaded automatically into `../data/` on the first run of
`load_cifar2()`. Both notebooks set `torch.manual_seed(...)` and
`torch.set_default_dtype(torch.double)` for reproducibility — the double precision is what
makes the `train` vs. `train_manual_update` comparison match to `1e-8`.

## Note on tooling

ChatGPT was used as a debugging aid for a few utility functions (the manual SGD update,
the plotting helpers, and best-state tracking in `evaluate_model`). These are marked with
comments in [gradient_descent.ipynb](gradient_descent.ipynb).
