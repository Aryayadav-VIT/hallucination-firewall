"""Safe model loading and validation module."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

from src.config import BASE_DIR, MODELS_DIR, EXPECTED_FEATURES

def find_model_path() -> Path:
    """Dynamically discover the model bundle file across standard project directories."""
    candidate_names = [
        "hallucination_firewall_bundle.joblib",
        "hallucination_firewall_corrected_bundle.joblib",
        "hallucination_firewall_corrected_bundle (3).joblib",
    ]

    search_dirs = [
        MODELS_DIR,
        BASE_DIR,
        BASE_DIR.parent / "models",
        BASE_DIR.parent,
    ]

    for d in search_dirs:
        for name in candidate_names:
            p = d / name
            if p.is_file():
                return p

    for d in search_dirs:
        if d.is_dir():
            matches = list(d.glob("*bundle*.joblib"))
            if matches:
                return matches[0]

    return MODELS_DIR / "hallucination_firewall_bundle.joblib"


def validate_bundle(bundle: Any) -> Dict[str, Any]:
    """Validate that the loaded bundle matches expected schema, features, and model class."""
    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary.")

    required_keys = [
        "model",
        "feature_columns",
        "threshold",
        "clip_model_name",
        "clip_pretrained",
        "target_definition",
    ]

    missing = [key for key in required_keys if key not in bundle]
    if missing:
        raise ValueError(f"Missing required model bundle keys: {missing}")

    feature_columns = list(bundle["feature_columns"])
    if feature_columns != EXPECTED_FEATURES:
        raise ValueError(
            f"Feature schema mismatch.\nExpected: {EXPECTED_FEATURES}\nFound: {feature_columns}"
        )

    model = bundle["model"]
    model_count = getattr(model, "n_features_in_", None)
    if model_count is not None and model_count != len(EXPECTED_FEATURES):
        raise ValueError(
            f"Classifier expects {model_count} features, but {len(EXPECTED_FEATURES)} are defined."
        )

    return bundle


# Internal cache for non-Streamlit execution (tests / CLI)
_BUNDLE_CACHE: Dict[str, Dict[str, Any]] = {}

def load_bundle(model_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate the model bundle from disk with caching."""
    path = Path(model_path) if model_path else find_model_path()
    path_str = str(path.resolve())

    if not path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at: {path_str}.\n"
            f"Searched in: {MODELS_DIR} and {BASE_DIR}"
        )

    # Check cache
    mtime = path.stat().st_mtime
    cache_key = f"{path_str}:{mtime}"
    if cache_key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[cache_key]

    raw_bundle = joblib.load(path)
    bundle = validate_bundle(raw_bundle)
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle
