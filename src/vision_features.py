"""Vision and multimodal feature extraction module using CLIP."""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
import open_clip

from src.config import CLIP_MODEL_NAME, CLIP_PRETRAINED

# Global cache for the vision model in memory
_VISION_MODEL_CACHE: Dict[str, Tuple[Any, Any, Any, str]] = {}


def load_clip_model(
    model_name: str = CLIP_MODEL_NAME,
    pretrained: str = CLIP_PRETRAINED
) -> Tuple[Any, Any, Any, str]:
    """Load and cache the CLIP model, preprocessor, and tokenizer."""
    cache_key = f"{model_name}:{pretrained}"
    if cache_key in _VISION_MODEL_CACHE:
        return _VISION_MODEL_CACHE[cache_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    loaded = (model, preprocess, tokenizer, device)
    _VISION_MODEL_CACHE[cache_key] = loaded
    return loaded


def object_prompts(obj: str) -> Tuple[List[str], List[str]]:
    """Generate positive and negative object presence prompt templates."""
    clean_obj = str(obj).strip().lower()
    positive = [
        f"an image containing {clean_obj}",
        f"a picture of {clean_obj}",
        f"a photo with {clean_obj}",
        f"{clean_obj} visible in the image",
    ]
    negative = [
        f"an image without {clean_obj}",
        f"a picture without {clean_obj}",
        f"a photo with no {clean_obj}",
        f"no {clean_obj} is visible in the image",
    ]
    return positive, negative


def extract_clip_similarities(
    image: Image.Image,
    texts: List[str],
    model_name: str = CLIP_MODEL_NAME,
    pretrained: str = CLIP_PRETRAINED,
) -> np.ndarray:
    """Compute cosine similarities between an image and a list of text strings."""
    model, preprocess, tokenizer, device = load_clip_model(model_name, pretrained)

    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image_tensor)
        text_embeddings = model.encode_text(tokens)

        # L2 Normalization
        image_embedding = image_embedding / image_embedding.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        text_embeddings = text_embeddings / text_embeddings.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        similarities = (image_embedding @ text_embeddings.T)[0].float().cpu().numpy()

    return similarities


def calculate_8_features(
    image: Image.Image,
    question: str,
    answer: str,
    target_object: str,
    claim_present: int,
) -> Dict[str, float]:
    """
    Calculate the exact 8 features expected by the trained SVM firewall model:
    1. image_question_similarity
    2. image_answer_similarity
    3. multimodal_grounding_score
    4. object_positive_similarity
    5. object_negative_similarity
    6. object_presence_margin
    7. answer_claim_present
    8. claim_evidence_alignment
    """
    positive_prompts, negative_prompts = object_prompts(target_object)
    texts = [question, answer] + positive_prompts + negative_prompts

    similarities = extract_clip_similarities(image, texts)

    image_question = float(similarities[0])
    image_answer = float(similarities[1])
    positive_similarity = float(np.mean(similarities[2:6]))
    negative_similarity = float(np.mean(similarities[6:10]))
    presence_margin = float(positive_similarity - negative_similarity)

    claim_sign = 1.0 if claim_present == 1 else -1.0

    features = {
        "image_question_similarity": image_question,
        "image_answer_similarity": image_answer,
        "multimodal_grounding_score": (image_question + image_answer) / 2.0,
        "object_positive_similarity": positive_similarity,
        "object_negative_similarity": negative_similarity,
        "object_presence_margin": presence_margin,
        "answer_claim_present": float(claim_present),
        "claim_evidence_alignment": float(claim_sign * presence_margin),
    }

    return features


def evaluate_contrastive_attribute(
    image: Image.Image,
    target_object: str,
    claimed_attribute: str,
    candidate_attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Evaluate an attribute (e.g., color) contrastively against alternative attributes."""
    if candidate_attributes is None:
        candidate_attributes = [
            "red", "blue", "green", "yellow", "black", "white", "silver", "gray", "brown", "orange"
        ]

    claimed = claimed_attribute.lower().strip()
    all_attrs = [claimed] + [a for a in candidate_attributes if a.lower() != claimed]

    texts = [f"a picture of a {attr} {target_object}" for attr in all_attrs]
    sims = extract_clip_similarities(image, texts)

    top_idx = int(np.argmax(sims))
    best_attr = all_attrs[top_idx]
    claimed_sim = float(sims[0])
    best_sim = float(sims[top_idx])

    return {
        "claimed_attribute": claimed,
        "best_matching_attribute": best_attr,
        "claimed_similarity": claimed_sim,
        "best_similarity": best_sim,
        "margin": float(claimed_sim - (sims[1] if len(sims) > 1 else claimed_sim)),
        "is_top_match": (top_idx == 0)
    }
