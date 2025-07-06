"""
Model persistence and storage management for ML insights engine.

Handles model serialization, versioning, and compressed storage for production deployment.
"""

import gzip
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...core.analytics_architecture import MLModelType, ModelId

if TYPE_CHECKING:
    from ..ml_insights_engine import MLModel
from ...core.errors import AnalyticsError


class ModelStorageError(AnalyticsError):
    """Model storage related errors."""

    pass


class ModelStorage:
    """Production model storage with versioning and compression."""

    def __init__(self, storage_path: str = "models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "model_metadata.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load model metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def save_model(self, model: "MLModel", version: str = "latest") -> str:
        """Save model with compression and versioning."""
        try:
            model_key = f"{model.model_type.value}_{model.model_id}"
            version_key = f"{model_key}_v{version}"

            # Create model directory
            model_dir = self.storage_path / model_key
            model_dir.mkdir(exist_ok=True)

            # Save model file with compression
            model_file = model_dir / f"{version_key}.pkl.gz"
            with gzip.open(model_file, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update metadata
            self.metadata[model_key] = {
                "model_type": model.model_type.value,
                "model_id": model.model_id,
                "versions": self.metadata.get(model_key, {}).get("versions", {}),
                "latest_version": version,
                "created_at": datetime.now(UTC).isoformat(),
            }

            # Version metadata
            self.metadata[model_key]["versions"][version] = {
                "file_path": str(model_file),
                "saved_at": datetime.now(UTC).isoformat(),
                "model_accuracy": getattr(model, "model_accuracy", 0.0),
                "training_data_size": getattr(model, "training_data_size", 0),
                "compressed_size_bytes": model_file.stat().st_size,
            }

            self._save_metadata()
            return str(model_file)

        except Exception as e:
            raise ModelStorageError(f"Failed to save model {model.model_id}: {e}")

    def load_model(
        self, model_type: MLModelType, model_id: ModelId, version: str = "latest"
    ) -> "MLModel":
        """Load model from compressed storage."""
        try:
            model_key = f"{model_type.value}_{model_id}"

            if model_key not in self.metadata:
                raise ModelStorageError(f"Model {model_key} not found")

            model_meta = self.metadata[model_key]

            # Get version to load
            if version == "latest":
                version = model_meta["latest_version"]

            if version not in model_meta["versions"]:
                raise ModelStorageError(
                    f"Version {version} not found for model {model_key}"
                )

            version_meta = model_meta["versions"][version]
            model_file = Path(version_meta["file_path"])

            if not model_file.exists():
                raise ModelStorageError(f"Model file not found: {model_file}")

            # Load compressed model
            with gzip.open(model_file, "rb") as f:
                model = pickle.load(f)

            return model

        except Exception as e:
            raise ModelStorageError(f"Failed to load model {model_id}: {e}")

    def list_models(self) -> list[dict[str, Any]]:
        """List all stored models with metadata."""
        models = []
        for model_key, model_meta in self.metadata.items():
            models.append(
                {
                    "model_key": model_key,
                    "model_type": model_meta["model_type"],
                    "model_id": model_meta["model_id"],
                    "latest_version": model_meta["latest_version"],
                    "versions": list(model_meta["versions"].keys()),
                    "created_at": model_meta["created_at"],
                }
            )
        return models

    def delete_model(
        self, model_type: MLModelType, model_id: ModelId, version: str | None = None
    ) -> bool:
        """Delete model or specific version."""
        try:
            model_key = f"{model_type.value}_{model_id}"

            if model_key not in self.metadata:
                return False

            if version is None:
                # Delete entire model
                model_dir = self.storage_path / model_key
                if model_dir.exists():
                    import shutil

                    shutil.rmtree(model_dir)
                del self.metadata[model_key]
            else:
                # Delete specific version
                model_meta = self.metadata[model_key]
                if version in model_meta["versions"]:
                    version_meta = model_meta["versions"][version]
                    model_file = Path(version_meta["file_path"])
                    if model_file.exists():
                        model_file.unlink()
                    del model_meta["versions"][version]

                    # Update latest version if needed
                    if model_meta["latest_version"] == version:
                        remaining_versions = list(model_meta["versions"].keys())
                        if remaining_versions:
                            model_meta["latest_version"] = remaining_versions[-1]
                        else:
                            del self.metadata[model_key]

            self._save_metadata()
            return True

        except Exception as e:
            raise ModelStorageError(f"Failed to delete model {model_id}: {e}")

    def get_model_info(
        self, model_type: MLModelType, model_id: ModelId
    ) -> dict[str, Any]:
        """Get detailed model information."""
        model_key = f"{model_type.value}_{model_id}"

        if model_key not in self.metadata:
            raise ModelStorageError(f"Model {model_key} not found")

        return self.metadata[model_key]

    def cleanup_old_versions(self, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping only the most recent."""
        cleaned_count = 0

        for model_key, model_meta in self.metadata.items():
            versions = model_meta["versions"]
            if len(versions) > keep_versions:
                # Sort versions by save date
                sorted_versions = sorted(
                    versions.items(), key=lambda x: x[1]["saved_at"], reverse=True
                )

                # Keep only the most recent versions
                versions_to_keep = [v[0] for v in sorted_versions[:keep_versions]]
                versions_to_delete = [v[0] for v in sorted_versions[keep_versions:]]

                for version in versions_to_delete:
                    try:
                        version_meta = versions[version]
                        model_file = Path(version_meta["file_path"])
                        if model_file.exists():
                            model_file.unlink()
                        del versions[version]
                        cleaned_count += 1
                    except Exception:
                        continue

        if cleaned_count > 0:
            self._save_metadata()

        return cleaned_count
