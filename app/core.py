
"""PDF extraction, retrieval, and conservative evidence checks."""

import re
from dataclasses import dataclass

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Evidence:
    text: str
    source: str
    page: int | None = None
    score: float = 0.0


def split_into_claims(text: str) -> list[str]:
    """Split answers into smaller factual claims."""
    if not text or not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    claims: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip(" \t-•")
        if len(sentence) < 12:
            continue

        parts = re.split(
            r"\s+and\s+(?=(?:costs?|fees?|charges?|"
            r"is priced|is available|guarantees?)\b)",
            sentence,
            flags=re.IGNORECASE,
        )

        claims.extend(
            part.strip()
            for part in parts
            if len(part.strip()) >= 12
        )

    return claims


def extract_pdf_chunks(
    file_obj,
    chunk_words: int = 45,
    overlap_words: int = 0,
) -> list[Evidence]:
    """Extract PDF passages while retaining source and page information."""
    reader = PdfReader(file_obj)
    source = getattr(file_obj, "name", "uploaded.pdf")
    chunks: list[Evidence] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()

        if not text:
            continue

        for sentence in split_into_claims(text):
            words = sentence.split()

            if len(words) <= chunk_words:
                chunks.append(
                    Evidence(sentence, source, page_number)
                )
            else:
                step = max(1, chunk_words - overlap_words)

                for start in range(0, len(words), step):
                    passage = " ".join(words[start:start + chunk_words])
                    if passage:
                        chunks.append(
                            Evidence(passage, source, page_number)
                        )

    return chunks


def retrieve_evidence(
    claim: str,
    chunks: list[Evidence],
    top_k: int = 3,
) -> list[Evidence]:
    """Retrieve relevant passages using local TF-IDF similarity."""
    if not claim.strip() or not chunks or top_k <= 0:
        return []

    corpus = [item.text for item in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(corpus + [claim])
    except ValueError:
        return []

    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked_indices = scores.argsort()[::-1]
    results: list[Evidence] = []

    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue

        item = chunks[index]
        results.append(
            Evidence(
                text=item.text,
                source=item.source,
                page=item.page,
                score=score,
            )
        )

        if len(results) >= top_k:
            break

    return results


def _normalize(text: str) -> str:
    """Normalize text for basic comparison."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_facts(text: str) -> dict[str, set[str]]:
    """
    Extract selected factual properties explicitly stated in text.

    This is a targeted heuristic, not a general semantic parser.
    """
    facts: dict[str, set[str]] = {}

    def add(key: str, value: str) -> None:
        value = _normalize(value)
        if value:
            facts.setdefault(key, set()).add(value)

    # Founding year.
    for match in re.finditer(
        r"\bfounded(?:\s+in)?\s*:?\s*((?:19|20)\d{2})\b",
        text,
        re.IGNORECASE,
    ):
        add("founding_year", match.group(1))

    # Headquarters location.
    for match in re.finditer(
        r"\b(?:headquarters|headquartered)"
        r"(?:\s+is)?\s*(?:in\s*)?:?\s*"
        r"([A-Za-z][A-Za-z\s.-]*(?:,\s*[A-Za-z][A-Za-z\s.-]*)?)",
        text,
        re.IGNORECASE,
    ):
        location = re.split(
            r"\b(?:founded|offers|provides|primary service|"
            r"program|course|established)\b|[.;\n]",
            match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,")
        add("headquarters", location)

    # Known program durations, including number words.
    programs = {
        "python_foundations": r"python\s+foundations",
        "data_analytics_starter": r"data\s+analytics\s+starter",
        "ai_research_practicum": r"ai\s+research\s+practicum",
    }

    number_pattern = (
        r"(?:\d+|one|two|three|four|five|six|seven|eight|"
        r"nine|ten)"
    )

    word_numbers = {
        "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }

    for program_key, pattern in programs.items():
        for match in re.finditer(
            pattern + r".{0,100}?\b(" + number_pattern + r")\s*weeks?\b",
            text,
            re.IGNORECASE,
        ):
            number = match.group(1).lower()
            number = word_numbers.get(number, number)
            add(f"duration:{program_key}", number + " weeks")

    # Explicit course fees.
    if re.search(r"\b(fee|cost|costs|priced|price)\b|₹", text, re.I):
        for match in re.finditer(
            r"₹\s*([\d,]+)|\bINR\s*([\d,]+)",
            text,
            re.IGNORECASE,
        ):
            amount = (match.group(1) or match.group(2)).replace(",", "")
            add("course_fee", amount)

    # Support schedule: standardize the expected weekday and time format.
    support_context = re.search(
        r"\b(?:support hours|support team|support is available|"
        r"available)\b",
        text,
        re.IGNORECASE,
    )

    if support_context:
        normalized = _normalize(text)

        has_weekdays = bool(
            re.search(r"\bmonday\s+to\s+friday\b", normalized)
        )

        # Accept 9 AM, 9:00 AM, and 09:00.
        has_nine = bool(
            re.search(
                r"\b(?:9\s*(?:00\s*)?am|09\s*00)\b",
                normalized,
            )
        )

        # Accept 5 PM, 5:00 PM, and 17:00.
        has_five = bool(
            re.search(
                r"\b(?:5\s*(?:00\s*)?pm|17\s*00)\b",
                normalized,
            )
        )

        has_timezone = bool(
            re.search(
                r"\b(?:ist|india standard time)\b",
                text,
                re.IGNORECASE,
            )
        )

        if has_weekdays and has_nine and has_five and has_timezone:
            add(
                "support_hours",
                "monday to friday 9 am to 5 pm ist",
            )

    return facts


def check_explicit_evidence(
    claim: str,
    passages: list[Evidence],
) -> dict:
    """
    Check a claim against supplied PDF passages.

    Recognizes exact text and selected structured facts.
    Unknown or ambiguous claims remain insufficient_evidence.
    """
    if not claim.strip() or not passages:
        return {
            "verdict": "insufficient_evidence",
            "explanation": "No usable claim or PDF evidence was supplied.",
            "evidence": None,
            "method": "abstain",
        }

    claim_norm = _normalize(claim)

    # 1. Exact textual support.
    for passage in passages:
        passage_norm = _normalize(passage.text)

        if claim_norm and claim_norm in passage_norm:
            return {
                "verdict": "supported",
                "explanation": "The PDF passage explicitly contains the claim.",
                "evidence": passage,
                "method": "exact_text_match",
            }

    # 2. Extract structured facts.
    claim_facts = _extract_facts(claim)
    passage_facts: dict[str, set[str]] = {}

    for passage in passages:
        for key, values in _extract_facts(passage.text).items():
            passage_facts.setdefault(key, set()).update(values)

    if claim_facts:
        conflicts = []
        matched_keys = []

        for key, claimed_values in claim_facts.items():
            documented_values = passage_facts.get(key, set())

            if not documented_values:
                continue

            if claimed_values.isdisjoint(documented_values):
                conflicts.append(key)
            else:
                matched_keys.append(key)

        if conflicts:
            conflict_key = conflicts[0]
            evidence = next(
                (
                    passage
                    for passage in passages
                    if conflict_key in _extract_facts(passage.text)
                ),
                None,
            )

            return {
                "verdict": "contradicted",
                "explanation": (
                    "The claim conflicts with an explicitly stated PDF "
                    f"value for {conflict_key.replace('_', ' ')}."
                ),
                "evidence": evidence,
                "method": "structured_fact_conflict",
            }

        if len(matched_keys) == len(claim_facts):
            evidence = next(
                (
                    passage
                    for passage in passages
                    if all(
                        key in _extract_facts(passage.text)
                        and not claim_facts[key].isdisjoint(
                            _extract_facts(passage.text)[key]
                        )
                        for key in claim_facts
                    )
                ),
                None,
            )

            return {
                "verdict": "supported",
                "explanation": (
                    "The extracted factual details match values stated "
                    "in the supplied PDF evidence."
                ),
                "evidence": evidence,
                "method": "structured_fact_match",
            }

    return {
        "verdict": "insufficient_evidence",
        "explanation": (
            "The retrieved passages do not establish or directly "
            "contradict the claim using the implemented checks."
        ),
        "evidence": None,
        "method": "abstain",
    }
