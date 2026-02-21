"""
Lightweight classifier for Brightness-based Lightweight Detector (BLD).

Uses extracted brightness features to classify images as real vs. AI-generated.
Intentionally simple — the key insight is in the features, not the model.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import json


class BLDClassifier:
    """
    Lightweight binary classifier for AI-generated image detection.

    Supports:
    - Logistic Regression (default, most lightweight)
    - Small MLP (2-layer, for slightly better accuracy)
    """

    def __init__(self, model_type: str = "logistic", hidden_dim: int = 64):
        """
        Args:
            model_type: 'logistic' or 'mlp'
            hidden_dim: Hidden layer size for MLP (ignored for logistic)
        """
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.model = None
        self.scaler = None

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Train the classifier.

        Args:
            features: NxD feature matrix.
            labels: N-length binary labels (0 = real, 1 = AI-generated).
            val_features: Optional validation features.
            val_labels: Optional validation labels.

        Returns:
            Training metrics dict.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Standardize features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(features)

        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train, labels)

        # Training metrics
        train_pred = self.model.predict(X_train)
        train_prob = self.model.predict_proba(X_train)[:, 1]
        metrics = {
            "train_acc": accuracy_score(labels, train_pred),
            "train_auc": roc_auc_score(labels, train_prob),
        }

        # Validation metrics
        if val_features is not None and val_labels is not None:
            X_val = self.scaler.transform(val_features)
            val_pred = self.model.predict(X_val)
            val_prob = self.model.predict_proba(X_val)[:, 1]
            metrics["val_acc"] = accuracy_score(val_labels, val_pred)
            metrics["val_auc"] = roc_auc_score(val_labels, val_prob)

        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary labels (0 = real, 1 = AI-generated)."""
        X = self.scaler.transform(features)
        return self.model.predict(X)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability of being AI-generated."""
        X = self.scaler.transform(features)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.pkl")
        joblib.dump(self.scaler, path / "scaler.pkl")
        with open(path / "config.json", "w") as f:
            json.dump({
                "model_type": self.model_type,
                "hidden_dim": self.hidden_dim,
            }, f)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        import joblib
        path = Path(path)
        self.model = joblib.load(path / "model.pkl")
        self.scaler = joblib.load(path / "scaler.pkl")
        with open(path / "config.json") as f:
            config = json.load(f)
            self.model_type = config["model_type"]
            self.hidden_dim = config["hidden_dim"]

    def param_count(self) -> int:
        """Return approximate number of trainable parameters."""
        if self.model_type == "logistic":
            # weights + bias
            return self.model.coef_.size + self.model.intercept_.size
        elif self.model_type == "mlp":
            total = 0
            for coef in self.model.coefs_:
                total += coef.size
            for bias in self.model.intercepts_:
                total += bias.size
            return total
        return 0
