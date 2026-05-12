import torch

from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.linear_model import LogisticRegression as SKLearnLR
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Probes
#
# A Probe holds the trained parameters for a single layer and exposes
# direction, score(), and predict(). score() returns raw logits
# (positive = positive class); predict() thresholds at zero.
# ---------------------------------------------------------------------------

class Probe(ABC):
    @property
    @abstractmethod
    def direction(self) -> torch.Tensor:
        """Unit vector representing the probe's linear direction. Shape (d_model,)."""
        ...

    @abstractmethod
    def score(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: (n, d_model) float tensor.

        Returns:
            (n,) float tensor of raw scores. Positive values indicate the
            positive class.
        """
        ...

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: (n, d_model) float tensor.

        Returns:
            (n,) bool tensor of predicted labels.
        """
        return self.score(activations) > 0

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the probe to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Probe":
        """Load a previously saved probe from disk."""
        ...


class DifferenceOfMeansProbe(Probe):
    """
    Linear probe whose direction is the normalized difference of class means.
    """

    def __init__(self, direction: torch.Tensor, threshold: float):
        self._direction = direction  # (d_model,) unit vector
        self.threshold = threshold   # midpoint of projected class means

    @property
    def direction(self) -> torch.Tensor:
        return self._direction

    def score(self, activations: torch.Tensor) -> torch.Tensor:
        return activations.float() @ self._direction.float() - self.threshold

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "type": "dom",
                "direction": self._direction,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "DifferenceOfMeansProbe":
        state = torch.load(path, weights_only=True)

        return cls(direction=state["direction"], threshold=state["threshold"])


class LogisticRegressionProbe(Probe):
    """
    Linear probe trained with logistic regression.

    When standard scaling was used at training time, scaler_mean and
    scaler_std are stored here and applied inside score() before the dot
    product.

    Parameters set by LogisticRegressionTrainer — do not construct directly.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        bias: float,
        scaler_mean: torch.Tensor | None = None,
        scaler_std: torch.Tensor | None = None,
    ):
        self.weights = weights          # (d_model,) unnormalized
        self.bias = bias
        self.scaler_mean = scaler_mean  # (d_model,) or None
        self.scaler_std = scaler_std    # (d_model,) or None

    @property
    def direction(self) -> torch.Tensor:
        norm = self.weights.norm()
        return self.weights / norm.clamp(min=1e-8)

    def _scale(self, activations: torch.Tensor) -> torch.Tensor:
        if self.scaler_mean is None:
            return activations

        return (activations - self.scaler_mean.to(activations.device)) / self.scaler_std.to(activations.device)

    def score(self, activations: torch.Tensor) -> torch.Tensor:
        x = self._scale(activations.float())
        return x @ self.weights.float() + self.bias

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "type": "lr",
                "weights": self.weights,
                "bias": self.bias,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionProbe":
        state = torch.load(path, weights_only=True)

        return cls(
            weights=state["weights"],
            bias=state["bias"],
            scaler_mean=state["scaler_mean"],
            scaler_std=state["scaler_std"],
        )


# ---------------------------------------------------------------------------
# Trainers
#
# A ProbeTrainer fits a Probe from (activations, labels) for one layer.
# fit_layers() is a convenience wrapper that calls fit() per layer and
# returns a dict keyed by layer index.
# ---------------------------------------------------------------------------

class ProbeTrainer(ABC):
    @abstractmethod
    def fit(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
    ) -> Probe:
        """
        Args:
            activations: (n, d_model) float tensor.
            labels:      (n,) bool tensor. True = positive class.

        Returns:
            A trained Probe for this layer.
        """
        ...

    def fit_layers(
        self,
        act_dict: dict[int, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[int, Probe]:
        """
        Train one probe per layer.

        Args:
            act_dict: dict mapping layer index -> (n, d_model) activations.
            labels:   (n,) bool tensor shared across layers.

        Returns:
            dict mapping layer index -> trained Probe.
        """
        return {layer: self.fit(acts, labels) for layer, acts in act_dict.items()}


class DifferenceOfMeansTrainer(ProbeTrainer):
    """
    Trains a probe whose direction is mean(positives) - mean(negatives),
    normalized to unit length. The threshold is set at the midpoint of the
    two projected class means so that score() > 0 predicts positive.
    """

    def fit(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
    ) -> DifferenceOfMeansProbe:
        acts = activations.float()
        pos_mean = acts[labels].mean(dim=0)
        neg_mean = acts[~labels].mean(dim=0)

        direction = pos_mean - neg_mean
        direction = direction / direction.norm()

        pos_proj = (acts[labels] @ direction).mean()
        neg_proj = (acts[~labels] @ direction).mean()
        threshold = ((pos_proj + neg_proj) / 2).item()

        return DifferenceOfMeansProbe(direction, threshold)


class LogisticRegressionTrainer(ProbeTrainer):
    """
    Trains a probe using sklearn's LogisticRegression.

    Args:
        C:        Inverse regularization strength (sklearn convention).
        max_iter: Maximum solver iterations.
        solver:   sklearn solver name.
        scale:    If True, apply StandardScaler to activations before fitting.
                  Scaler parameters are stored on the probe and applied at
                  inference time.
    """

    def __init__(
        self,
        C: float = 0.1,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        scale: bool = False,
    ):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.scale = scale

    def fit(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
    ) -> LogisticRegressionProbe:
        X = activations.float().cpu().numpy()
        y = labels.cpu().numpy()

        scaler_mean = scaler_std = None

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
            scaler_std = torch.tensor(scaler.scale_, dtype=torch.float32)

        clf = SKLearnLR(C=self.C, max_iter=self.max_iter, solver=self.solver)
        clf.fit(X, y)

        weights = torch.tensor(clf.coef_[0], dtype=torch.float32)
        bias = float(clf.intercept_[0])

        return LogisticRegressionProbe(weights, bias, scaler_mean, scaler_std)