"""Core evidence-aware verification dispatcher for multimodal hallucination categories."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from PIL import Image

from src.config import (
    HallucinationCategory,
    VerificationStatus,
    DEFAULT_THRESHOLD,
    EXPECTED_FEATURES
)
from src.question_parser import ParsedClaim
from src.vision_features import (
    calculate_8_features,
    evaluate_contrastive_attribute,
    extract_clip_similarities
)
from src.model_loader import load_bundle


@dataclass
class VerificationResult:
    status: str
    category: str
    target_object: Optional[str] = None
    method_used: str = ""
    risk_score: Optional[float] = None
    threshold: Optional[float] = None
    evidence_summary: str = ""
    limitations: List[str] = field(default_factory=list)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    is_live_prediction: bool = True


def verify_claim(
    image: Image.Image,
    claim: ParsedClaim,
    threshold: float = DEFAULT_THRESHOLD,
    model_bundle: Optional[Dict[str, Any]] = None,
) -> VerificationResult:
    """
    Central verification pipeline dispatcher.
    Evaluates the claim against visual evidence using the appropriate method.
    """
    if model_bundle is None:
        model_bundle = load_bundle()

    # 1. External / Unverifiable Speculation
    if claim.category == HallucinationCategory.UNSUPPORTED_SPECULATION:
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED_CHECK_TYPE,
            category=claim.category,
            target_object=claim.target_object,
            method_used="Domain Constraint / Pixel-Evidence Filter",
            risk_score=None,
            threshold=threshold,
            evidence_summary="The claim involves non-visual metadata (monetary cost, exact age, calendar date, or private identity) that cannot be determined from pixel content alone.",
            limitations=[
                "External and non-visual claims are fundamentally ungroundable in image pixels.",
                "No vision model can determine purchase price, exact biographical age, or private identity without external databases."
            ],
            detailed_metrics={"verifiable_from_pixels": False}
        )

    # 2. Object Count
    if claim.category == HallucinationCategory.COUNTING:
        # Compute baseline similarity as an indicator
        sim = float(extract_clip_similarities(image, [claim.raw_answer])[0])
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED_CHECK_TYPE,
            category=claim.category,
            target_object=claim.target_object,
            method_used="Experimental Heuristic (Counting Limitation Flag)",
            risk_score=None,
            threshold=threshold,
            evidence_summary=f"Claimed count: {claim.claimed_count or 'unspecified'} {claim.target_object or 'items'}. CLIP image-answer similarity is {sim:.4f}.",
            limitations=[
                "Standard vision-language models (CLIP) do not possess object counting heads or instance segmentation.",
                "CLIP similarity cannot reliably verify exact counts (e.g. distinguishing 2 vs 3 items).",
                "True count verification requires dedicated object detection (e.g. YOLO, DETR) or instance segmentation."
            ],
            detailed_metrics={
                "claimed_count": claim.claimed_count,
                "clip_similarity": sim,
                "counting_model_available": False
            }
        )

    # 3. Spatial Relation
    if claim.category == HallucinationCategory.SPATIAL_RELATION:
        subj = claim.target_object or "object"
        rel = claim.relation or "near"
        ref = claim.reference_object or "other object"
        
        pos_prompt = f"a photo of a {subj} {rel} the {ref}"
        neg_prompt = f"a photo of a {ref} {rel} the {subj}"
        sims = extract_clip_similarities(image, [pos_prompt, neg_prompt])
        diff = float(sims[0] - sims[1])

        status = VerificationStatus.INCONCLUSIVE
        if abs(diff) > 0.03:
            status = VerificationStatus.LIKELY_SUPPORTED if diff > 0 else VerificationStatus.POTENTIAL_HALLUCINATION

        return VerificationResult(
            status=status,
            category=claim.category,
            target_object=f"{subj} {rel} {ref}",
            method_used="Relational Prompt Comparison (Zero-Shot Heuristic)",
            risk_score=float(1.0 - sims[0]),
            threshold=threshold,
            evidence_summary=f"Relational contrast: '{pos_prompt}' ({sims[0]:.4f}) vs '{neg_prompt}' ({sims[1]:.4f}). Margin: {diff:+.4f}.",
            limitations=[
                "Vision-language models suffer from known 'compositionality blindness' (often unable to distinguish 'A on B' from 'B on A').",
                "Relational similarity is an experimental heuristic, not definitive geometric proof."
            ],
            detailed_metrics={
                "target_prompt_sim": float(sims[0]),
                "inverse_prompt_sim": float(sims[1]),
                "relational_margin": diff
            }
        )

    # 4. Visual Attribute (e.g. Color)
    if claim.category == HallucinationCategory.ATTRIBUTE and claim.target_attribute:
        obj = claim.target_object or "object"
        attr = claim.target_attribute
        attr_result = evaluate_contrastive_attribute(image, obj, attr)

        if attr_result["is_top_match"] and attr_result["margin"] >= 0.01:
            status = VerificationStatus.LIKELY_SUPPORTED
            ev = f"Claimed attribute '{attr}' matches visual evidence best (Similarity: {attr_result['claimed_similarity']:.4f}, Margin: {attr_result['margin']:+.4f})."
        elif not attr_result["is_top_match"]:
            status = VerificationStatus.POTENTIAL_HALLUCINATION
            ev = f"Contradiction: Visual evidence is more consistent with '{attr_result['best_matching_attribute']}' ({attr_result['best_similarity']:.4f}) than claimed '{attr}' ({attr_result['claimed_similarity']:.4f})."
        else:
            status = VerificationStatus.INCONCLUSIVE
            ev = f"Ambiguous attribute match: '{attr}' similarity is {attr_result['claimed_similarity']:.4f} with small margin."

        return VerificationResult(
            status=status,
            category=claim.category,
            target_object=f"{attr} {obj}",
            method_used="Contrastive Attribute Ranking (Zero-Shot CLIP Heuristic)",
            risk_score=float(1.0 - attr_result["claimed_similarity"]),
            threshold=threshold,
            evidence_summary=ev,
            limitations=[
                "Evaluated using whole-image CLIP prompt comparison across 10 common attribute descriptors.",
                "Does not isolate object bounding boxes; background colors may introduce noise."
            ],
            detailed_metrics=attr_result
        )

    # 5. Text / OCR
    if claim.category == HallucinationCategory.TEXT_READING:
        sim = float(extract_clip_similarities(image, [claim.raw_answer])[0])
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            category=claim.category,
            target_object="text",
            method_used="OCR Heuristic Check",
            risk_score=None,
            threshold=threshold,
            evidence_summary=f"Text verification requested. CLIP whole-image text similarity is {sim:.4f}.",
            limitations=[
                "Dedicated Optical Character Recognition (OCR) is required for reliable character-level verification.",
                "CLIP can recognize prominent signage concepts, but cannot reliably read dense, low-resolution, or rotated text."
            ],
            detailed_metrics={"text_similarity": sim, "dedicated_ocr_active": False}
        )

    # 6. Object Existence (Primary Trained SVM Domain)
    target_obj = claim.target_object or "object"
    claim_polarity = claim.claim_polarity if claim.claim_polarity is not None else 1

    features = calculate_8_features(
        image=image,
        question=claim.raw_query,
        answer=claim.raw_answer,
        target_object=target_obj,
        claim_present=claim_polarity,
    )

    feature_columns = model_bundle["feature_columns"]
    X = pd.DataFrame([[features[name] for name in feature_columns]], columns=feature_columns)

    classifier = model_bundle["model"]
    classes = list(classifier.classes_)

    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Model does not provide probability estimates.")

    probabilities = classifier.predict_proba(X)[0]
    # Class 1 = Hallucination / Contradiction candidate
    hallucination_risk = float(probabilities[classes.index(1)])
    support_probability = float(probabilities[classes.index(0)])
    margin = features["object_presence_margin"]

    # Evidence-aware thresholding
    if abs(margin) < 0.003:
        status = VerificationStatus.INCONCLUSIVE
        ev = (
            f"Object-presence evidence is ambiguous. CLIP positive-vs-negative prompt margin is nearly zero "
            f"({margin:+.4f}). SVM risk estimate is {hallucination_risk:.1%}."
        )
    elif hallucination_risk >= threshold:
        status = VerificationStatus.POTENTIAL_HALLUCINATION
        ev = (
            f"The candidate answer contradicts visual evidence with an SVM risk score of {hallucination_risk:.1%} "
            f"(threshold: {threshold:.2f}). Presence margin: {margin:+.4f}."
        )
    else:
        status = VerificationStatus.LIKELY_SUPPORTED
        ev = (
            f"The candidate answer is supported by visual evidence with an estimated support probability of {support_probability:.1%} "
            f"(risk: {hallucination_risk:.1%} vs threshold {threshold:.2f}). Presence margin: {margin:+.4f}."
        )

    return VerificationResult(
        status=status,
        category=HallucinationCategory.OBJECT_EXISTENCE,
        target_object=target_obj,
        method_used="Trained Multimodal SVM (8 CLIP-based features, trained on POPE benchmark)",
        risk_score=hallucination_risk,
        threshold=threshold,
        evidence_summary=ev,
        limitations=[
            "Trained specifically on POPE object-presence binary claims.",
            "Predictions are statistical confidence estimates, not definitive ground-truth object detection.",
            "Heuristic presence margin may be uncertain for heavily occluded or very small objects."
        ],
        detailed_metrics={
            **features,
            "support_probability": support_probability,
            "hallucination_risk": hallucination_risk,
        }
    )
