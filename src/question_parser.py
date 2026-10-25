"""Advanced deterministic question and claim parser for multimodal hallucination detection."""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.config import HallucinationCategory

# Meta words that indicate the container/medium, never target objects
META_WORDS = {
    "image", "picture", "photo", "photograph", "there", "visible", 
    "present", "anything", "something", "scene", "view", "frame",
    "background", "foreground", "canvas"
}

# Common plural-to-singular mappings for prompt consistency
PLURAL_MAP = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "cars": "car",
    "dogs": "dog",
    "cats": "cat",
    "birds": "bird",
    "phones": "phone",
    "cellphones": "phone",
    "smartphones": "phone",
    "bicycles": "bicycle",
    "bikes": "bicycle",
    "chairs": "chair",
    "tables": "table",
    "cups": "cup",
    "bottles": "bottle",
    "trees": "tree",
    "apples": "apple",
    "boxes": "box",
    "buses": "bus",
    "passengers": "passenger"
}

NUMBER_WORDS = {
    "zero": 0, "none": 0, "no": 0,
    "one": 1, "single": 1,
    "two": 2, "pair": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10
}

@dataclass
class ParsedClaim:
    category: str
    target_object: Optional[str] = None
    target_attribute: Optional[str] = None
    claimed_count: Optional[int] = None
    relation: Optional[str] = None
    reference_object: Optional[str] = None
    claim_polarity: Optional[int] = None  # 1 = positive, 0 = negative, None = ambiguous
    raw_query: str = ""
    raw_answer: str = ""
    is_supported_category: bool = True
    explanation: str = ""


def normalize_text(text: str) -> str:
    """Normalize whitespace, contractions, and punctuation."""
    t = str(text).strip()
    contractions = {
        r"\bthere's\b": "there is",
        r"\bcan't\b": "cannot",
        r"\bcannot\b": "cannot",
        r"\bdoesn't\b": "does not",
        r"\bdon't\b": "do not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bwhat's\b": "what is",
        r"\bit's\b": "it is",
    }
    for pattern, replacement in contractions.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_object_name(obj: Optional[str]) -> Optional[str]:
    """Clean extracted object names and strip determiners and meta words."""
    if not obj:
        return None
    o = obj.strip(" .?!,;:")
    # Strip leading determiners / quantifiers / negations with word boundaries
    o = re.sub(r"^\b(?:neither|another|some|not|any|the|one|no|an|a)\b\s*", "", o, flags=re.IGNORECASE).strip()
    # Strip trailing location phrases referencing the image container
    o = re.sub(r"\s+(?:in|on|inside|visible in)\s+(?:the|this|an?)\s+(?:image|picture|photo)$", "", o, flags=re.IGNORECASE).strip()
    # Strip trailing 'visible', 'present', 'there'
    o = re.sub(r"\s+\b(?:visible|present|there)\b$", "", o, flags=re.IGNORECASE).strip()
    
    if not o or o.lower() in META_WORDS:
        return None
    return o


def singularize(word: Optional[str]) -> str:
    """Convert common plural nouns to singular form for prompt consistency."""
    if not word:
        return "object"
    w = word.strip().lower()
    if w in PLURAL_MAP:
        return PLURAL_MAP[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3 and not w.endswith("ses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        return w[:-1]
    return w


def infer_claim_polarity(answer: str) -> Optional[int]:
    """Infer whether an answer makes a positive claim (1), negative claim (0), or is ambiguous (None)."""
    a = normalize_text(answer).lower()

    negative_patterns = [
        r"\bno\b", r"\bnot present\b", r"\bnot visible\b",
        r"\bdoes not contain\b", r"\bdoesn't contain\b",
        r"\bthere (?:is|are) no\b", r"\bthere is not\b", r"\bthere are not\b",
        r"\bwithout\b", r"\bcannot see\b", r"\bcan't see\b",
        r"\bdo not see\b", r"\bdon't see\b", r"\babsent\b",
        r"\bnone\b", r"\bzero\b", r"\bfalse\b", r"\bincorrect\b"
    ]

    for p in negative_patterns:
        if re.search(p, a):
            return 0

    positive_patterns = [
        r"^\s*yes\b", r"\bthere (?:is|are)\b", r"\bcontains?\b",
        r"\bshows?\b", r"\bvisible\b", r"\bi (?:can )?see\b",
        r"\bpresent\b", r"\btrue\b", r"\bcorrect\b"
    ]

    for p in positive_patterns:
        if re.search(p, a):
            return 1

    return None


def parse_question_and_answer(question: str, answer: str) -> ParsedClaim:
    """
    Parse a question and candidate answer into a structured ParsedClaim.
    Determines category, extracted entities, and claim polarity.
    """
    q_norm = normalize_text(question)
    a_norm = normalize_text(answer)
    q_clean = q_norm.strip(" ?.!")

    # 1. External / Non-visual Speculation Question
    speculation_patterns = [
        r"\b(?:how much|price|cost|worth|rupees|dollars|rs|inr|lakh|crore|\$)\b",
        r"\b(?:how old|age|date of birth)\b",
        r"\b(?:who is (?:this|the)|what is (?:the|his|her) name)\b",
        r"\b(?:when was (?:this|the photo) taken|what year|what date|yesterday|tomorrow)\b",
    ]
    for p in speculation_patterns:
        if re.search(p, q_clean, re.IGNORECASE):
            return ParsedClaim(
                category=HallucinationCategory.UNSUPPORTED_SPECULATION,
                raw_query=question,
                raw_answer=answer,
                is_supported_category=False,
                explanation="Non-visual metadata (price, exact age, identity, date/location) cannot be confirmed from image pixels alone."
            )

    # 2. Count Question
    count_match = re.search(r"^(?:how many|what is the count of|number of)\s+(.+?)(?:\s+(?:are|is|in|inside|visible).*)?$", q_clean, re.IGNORECASE)
    if count_match:
        raw_obj = clean_object_name(count_match.group(1))
        # Extract claimed number from answer
        claimed_num = None
        for word, val in NUMBER_WORDS.items():
            if re.search(rf"\b{word}\b", a_norm, re.IGNORECASE):
                claimed_num = val
                break
        if claimed_num is None:
            digit_match = re.search(r"\b(\d+)\b", a_norm)
            if digit_match:
                claimed_num = int(digit_match.group(1))

        return ParsedClaim(
            category=HallucinationCategory.COUNTING,
            target_object=singularize(raw_obj),
            claimed_count=claimed_num,
            claim_polarity=1 if (claimed_num is not None and claimed_num > 0) else 0,
            raw_query=question,
            raw_answer=answer,
            is_supported_category=False,
            explanation="Object counting requires dedicated object detection or regression. CLIP similarity alone cannot reliably count instances."
        )

    # 3. Attribute / Color Question
    color_patterns = [
        r"^(?:what color is|what is the color of)\s+(?:a|an|the)?\s*(.+)$",
        r"^(?:is|are)\s+(?:a|an|the)?\s*(.+?)\s+(red|blue|green|yellow|black|white|silver|gray|grey|brown|orange|pink|purple)$",
    ]
    for p in color_patterns:
        m = re.match(p, q_clean, re.IGNORECASE)
        if m:
            target_obj = clean_object_name(m.group(1))
            claimed_attr = m.group(2) if len(m.groups()) >= 2 else None
            if not claimed_attr:
                color_search = re.search(r"\b(red|blue|green|yellow|black|white|silver|gray|grey|brown|orange|pink|purple)\b", a_norm, re.IGNORECASE)
                if color_search:
                    claimed_attr = color_search.group(1).lower()

            return ParsedClaim(
                category=HallucinationCategory.ATTRIBUTE,
                target_object=singularize(target_obj),
                target_attribute=claimed_attr,
                claim_polarity=1 if claimed_attr else infer_claim_polarity(answer),
                raw_query=question,
                raw_answer=answer,
                is_supported_category=True,
                explanation="Visual attribute check: evaluated via contrastive attribute prompt comparison."
            )

    # 4. Text Reading / OCR Question
    text_match = re.search(r"\b(?:sign|text|word|read|say|written)\b", q_clean, re.IGNORECASE)
    if text_match and not re.search(r"\b(?:is there a|are there)\b", q_clean, re.IGNORECASE):
        return ParsedClaim(
            category=HallucinationCategory.TEXT_READING,
            claim_polarity=infer_claim_polarity(answer),
            raw_query=question,
            raw_answer=answer,
            is_supported_category=False,
            explanation="OCR verification: requires text reading or OCR engine."
        )

    # 5. Scene / Context Question
    scene_match = re.search(r"^(?:is this|does this show|is the photo taken in)\s+(?:a|an|the)?\s*(airport|beach|kitchen|bedroom|classroom|office|street|park|forest|hospital|restaurant|store|gym|living room)\b", q_clean, re.IGNORECASE)
    if scene_match:
        scene_type = scene_match.group(1)
        return ParsedClaim(
            category=HallucinationCategory.SCENE_CONTEXT,
            target_object=scene_type,
            claim_polarity=infer_claim_polarity(answer),
            raw_query=question,
            raw_answer=answer,
            is_supported_category=True,
            explanation="Scene classification: evaluated via zero-shot scene prompt matching."
        )

    # 6. Action / Activity Question
    action_match = re.search(r"^(?:is|are)\s+(?:a|an|the)?\s*(.+?)\s+([a-z]+ing)\b(?:\s+(.+))?$", q_clean, re.IGNORECASE)
    if action_match:
        subj = clean_object_name(action_match.group(1))
        action_verb = action_match.group(2)
        if subj and subj not in {"there", "it"}:
            return ParsedClaim(
                category=HallucinationCategory.ACTION_ACTIVITY,
                target_object=singularize(subj),
                target_attribute=action_verb,
                claim_polarity=infer_claim_polarity(answer),
                raw_query=question,
                raw_answer=answer,
                is_supported_category=True,
                explanation="Action verification: evaluated via contrastive action prompt matching."
            )

    # 7. Spatial Relation Question (Must reference TWO real objects, neither being meta-words like 'image')
    spatial_match = re.search(r"^(?:is|are)\s+(?:a|an|the)?\s*(.+?)\s+(on|in|under|behind|beside|next to|above|below|near)\s+(?:a|an|the)?\s*(.+)$", q_clean, re.IGNORECASE)
    if spatial_match:
        subj = clean_object_name(spatial_match.group(1))
        rel = spatial_match.group(2)
        ref_obj = clean_object_name(spatial_match.group(3))
        # Ensure neither subject nor reference object is a meta-word
        if subj and ref_obj and subj not in META_WORDS and ref_obj not in META_WORDS:
            return ParsedClaim(
                category=HallucinationCategory.SPATIAL_RELATION,
                target_object=singularize(subj),
                relation=rel,
                reference_object=singularize(ref_obj),
                claim_polarity=infer_claim_polarity(answer),
                raw_query=question,
                raw_answer=answer,
                is_supported_category=False,
                explanation="Spatial relation check: Vision-language models often struggle with spatial compositionality."
            )

    # 8. Object Existence (Primary domain of the trained SVM model)
    existence_patterns = [
        # "Is there a phone in the image?" / "Is there any phone visible?" / "Is there a phone?"
        r"^(?:is there|are there any|are there)\s+(?:\b(?:any|the|an|a)\b\s*)?(.+?)(?:\s+(?:in|inside|visible in|on)\s+(?:the|this|an?)\s+(?:image|picture|photo))?$",
        # "Can you see a dog?" / "Do you see a dog?"
        r"^(?:can you see|do you see|can we see)\s+(?:\b(?:any|the|an|a)\b\s*)?(.+?)(?:\s+(?:in|inside|visible in|on)\s+(?:the|this|an?)\s+(?:image|picture|photo))?$",
        # "Does the image contain/have/show a car?"
        r"^does\s+(?:the|this)\s+(?:image|picture|photo)\s+(?:contain|have|show|feature)\s+(?:\b(?:any|the|an|a)\b\s*)?(.+)$",
        # "Is a car present?" / "Is a phone visible?"
        r"^(?:is|are)\s+(?:\b(?:any|the|an|a)\b\s*)?(.+?)\s+(?:present|visible|there)(?:\s+(?:in|inside)\s+(?:the|this)\s+(?:image|picture|photo))?$",
        # Declarative assertions in question/prompt: "There is a dog in the image" / "There is no phone"
        r"^there\s+(?:is|are)\s+(?:\b(?:any|the|not|no|an|a)\b\s*)?(.+?)(?:\s+(?:in|inside|visible in)\s+(?:the|this)\s+(?:image|picture|photo))?$",
        # "The image contains no people"
        r"^(?:the|this)\s+(?:image|picture|photo)\s+(?:contains|shows|has)\s+(?:\b(?:any|the|not|no|an|a)\b\s*)?(.+)$",
    ]

    for p in existence_patterns:
        m = re.match(p, q_clean, re.IGNORECASE)
        if m:
            raw_obj = clean_object_name(m.group(1))
            if raw_obj:
                return ParsedClaim(
                    category=HallucinationCategory.OBJECT_EXISTENCE,
                    target_object=singularize(raw_obj),
                    claim_polarity=infer_claim_polarity(answer),
                    raw_query=question,
                    raw_answer=answer,
                    is_supported_category=True,
                    explanation="Object-presence candidate: fully evaluated by the 8-feature SVM model."
                )

    # Fallback heuristic: Try to find a noun phrase if question is simple
    fallback_words = [w for w in q_clean.split() if w.lower() not in META_WORDS and len(w) > 2]
    if fallback_words:
        candidate_obj = clean_object_name(fallback_words[-1])
        if candidate_obj:
            return ParsedClaim(
                category=HallucinationCategory.OBJECT_EXISTENCE,
                target_object=singularize(candidate_obj),
                claim_polarity=infer_claim_polarity(answer),
                raw_query=question,
                raw_answer=answer,
                is_supported_category=True,
                explanation="Object identified via flexible parsing heuristic."
            )

    # If completely unparseable
    return ParsedClaim(
        category=HallucinationCategory.OBJECT_EXISTENCE,
        target_object=None,
        claim_polarity=infer_claim_polarity(answer),
        raw_query=question,
        raw_answer=answer,
        is_supported_category=False,
        explanation="Could not identify a clear object, attribute, count, or relation from the question."
    )
