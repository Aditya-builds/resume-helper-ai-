"""
resume_scoring.py

Usage:
    - Set OPENAI_API_KEY environment variable or pass it to `score_resume()` directly.
    - Call score_resume(jd_text, resume_path, openai_api_key="...")

Outputs a dict with:
  - total_score, breakdown, missing_items, verdict, suggestions
"""

import os
import re
import math
import json
from typing import List, Dict, Tuple
import pdfplumber
import docx
import numpy as np
import openai
import spacy
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ---------- CONFIG / WEIGHTS ----------
WEIGHTS = {
    "keyword": 20,
    "semantic": 20,
    "experience": 25,
    "section": 15,
    "micro": 20
}

EMBEDDING_MODEL = "text-embedding-3-small"  # change if you prefer another model
SIMILARITY_MAX = 20  # mapping cosine similarity to 0..20

# thresholds / tuning
HIGH_FIT_THRESHOLD = 85
MEDIUM_FIT_THRESHOLD = 60

# heuristics
MAX_EXPERIENCE_YEARS_COUNTED = 25  # cap for experience points
EXPERIENCE_POINTS_PER_YEAR = WEIGHTS["experience"] / MAX_EXPERIENCE_YEARS_COUNTED

# load spaCy: try to load the small English model; if not available (Cloud restrictions),
# fall back to a lightweight blank English pipeline which is functional but has reduced NLP features.
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    try:
        # Fallback: create a blank English pipeline (no parser/ner by default)
        nlp = spacy.blank("en")
        # Optionally register basic components or warn that functionality is reduced
        print("Warning: 'en_core_web_sm' not available. Using spacy.blank('en') fallback with reduced features.")
    except Exception:
        nlp = None


# ---------- UTILITIES ----------

def extract_text_from_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def extract_text(path: str) -> str:
    file_path = path.lower()
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif file_path.endswith((".docx", ".doc")):
        return extract_text_from_docx(path)
    else:
        # assume plain text
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text_for_embedding(s: str, max_chars=2000) -> List[str]:
    """Split long text into chunks safe for embedding use."""
    s = s.strip()
    if len(s) <= max_chars:
        return [s]
    words = s.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len > max_chars:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def get_openai_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    # Prefer the explicit api_key parameter, fallback to the OPENAI_API_KEY env var
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key to get_openai_embeddings().")
    openai.api_key = api_key
    # join or call in batch - OpenAI python SDK supports embeddings.create with model and input list
    resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embs = [item.embedding for item in resp.data]
    return embs

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ---------- JD / KEYWORD EXTRACTION ----------

def extract_candidate_keywords_from_jd(jd_text: str, top_k=40) -> Dict[str, List[str]]:
    """
    Heuristic extraction:
      - Pull bullets/lines from JD
      - Use noun chunks + named entities + simple regex to build a list of keywords
      - Also return a short 'must_have' subset if the jd contains words like 'must', 'required'
    """
    jd_text = clean_text(jd_text)
    doc = nlp(jd_text)
    nouns = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
    ents = [ent.text.strip() for ent in doc.ents if len(ent.text.split()) <= 4]
    # words appearing frequently and longer than 2 chars
    words = re.findall(r"[A-Za-z0-9\+\#\-\_\.]{2,}", jd_text)
    words = [w for w in words if not w.isdigit()]
    freq = Counter(words)
    common_words = [w for w, _ in freq.most_common(100) if len(w) > 2]

    # heuristics for must-have: lines containing 'must have', 'required', 'min', 'should have'
    must_have = set()
    for line in jd_text.splitlines():
        low = line.lower()
        if any(k in low for k in ["must have", "must", "required", "minimum", "min", "should have"]):
            # extract words from line
            for token in re.findall(r"[A-Za-z0-9\+\#\-\_\.]{2,}", line):
                if len(token) > 2:
                    must_have.add(token)
    # fallback: top noun chunks and entities as keywords
    keywords = list(dict.fromkeys(nouns + ents + common_words))
    # cleanup: lowercase unique
    keywords = [k for k in keywords if len(k) > 1][:top_k]
    must_have_list = [k for k in list(must_have) if len(k) > 1]
    return {"keywords": keywords, "must_have": must_have_list}

# ---------- KEYWORD MATCHING ----------

def keyword_and_fuzzy_score(jd_keywords: List[str], resume_text: str, must_have: List[str]=None) -> Tuple[float, Dict]:
    """
    Score out of WEIGHTS['keyword'].
    Exact matches get more; fuzzy matches partial.
    Missing must_have items incur penalties.
    """
    if must_have is None:
        must_have = []

    resume_lower = resume_text.lower()
    hits = []
    fuzzy_hits = []
    for kw in jd_keywords:
        kw_l = kw.lower()
        if kw_l in resume_lower:
            hits.append(kw)
        else:
            # fuzzy check using rapidfuzz (ratio)
            score = fuzz.token_set_ratio(kw_l, resume_lower)
            if score >= 75:
                fuzzy_hits.append((kw, score))

    # score calculation
    max_pts = WEIGHTS["keyword"]
    # give exact matches heavier weight than fuzzy
    exact_pts_per = 1.2  # will be normalized below
    fuzzy_pts_per = 0.6

    # raw points
    raw = len(hits) * exact_pts_per + len(fuzzy_hits) * fuzzy_pts_per

    # normalize raw to max_pts by considering expected number of keywords (len(jd_keywords))
    expected = max(1, len(jd_keywords))
    normalized = raw / (expected * (exact_pts_per + fuzzy_pts_per) / 2) * max_pts
    normalized = min(normalized, max_pts)

    # heavy penalty if must_have missing: reduce score by 30% per missing important item (capped)
    missing_required = []
    for req in must_have:
        if req.lower() not in resume_lower:
            missing_required.append(req)
    if missing_required:
        # penalty proportional to count
        penalty_fraction = min(0.9, 0.30 * len(missing_required))  # don't zero out completely
        normalized = normalized * (1 - penalty_fraction)

    details = {
        "exact_matches": hits,
        "fuzzy_matches": [x[0] for x in fuzzy_hits],
        "missing_required": missing_required,
        "raw": raw,
        "normalized": round(normalized, 2),
        "max": max_pts
    }
    return normalized, details

# ---------- SEMANTIC SIMILARITY ----------

def semantic_similarity_score(jd_text: str, resume_text: str, api_key: str) -> Tuple[float, Dict]:
    """
    Use embeddings to compute cosine similarity, map to WEIGHTS['semantic'].
    If resume is long, embed chunks and average embeddings.
    """
    # chunk and get embeddings
    jd_chunks = chunk_text_for_embedding(jd_text)
    res_chunks = chunk_text_for_embedding(resume_text)

    jd_embs = get_openai_embeddings(jd_chunks, api_key)
    res_embs = get_openai_embeddings(res_chunks, api_key)

    jd_vec = np.mean(np.array(jd_embs), axis=0)
    res_vec = np.mean(np.array(res_embs), axis=0)

    sim = cosine_similarity(jd_vec, res_vec)  # -1..1
    # mostly 0..1 for text embeddings
    sim = max(0.0, sim)
    # map to 0..WEIGHTS['semantic'] (we scale linearly; you can change to non-linear)
    score = sim * WEIGHTS["semantic"]
    # cap
    score = min(score, WEIGHTS["semantic"])
    return round(score, 2), {"cosine_similarity": round(sim, 4), "mapped_score": round(score, 2), "max": WEIGHTS["semantic"]}

# ---------- EXPERIENCE PARSING & SCORING ----------

def extract_years_of_experience(resume_text: str, jd_keywords: List[str]) -> Tuple[float, Dict]:
    """
    Heuristic:
      - Find explicit patterns like 'X years', 'X yrs', 'from 2018 to 2022', '2018-2022'
      - Attribute years to relevant experience if those lines mention JD keywords.
      - Full-time experience > internships get more weight via keyword signals ('intern', 'internship')
    """
    text = resume_text
    years_score = 0.0
    details = {"found_years": [], "relevant_years_estimate": 0.0, "raw_matches": []}

    # pattern 1: 'X years'
    for m in re.finditer(r"(\d+)\s+(?:years|yrs|year)", text, flags=re.I):
        val = int(m.group(1))
        # examine context window
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        context = text[start:end].lower()
        relevant = any(kw.lower() in context for kw in jd_keywords)
        # penalize if 'intern' in context
        is_intern = bool(re.search(r"\bintern(ship)?\b", context, flags=re.I))
        score_years = val * (0.5 if is_intern else 1.0)
        details["raw_matches"].append({"match": m.group(0), "context": context[:120], "relevant": relevant, "intern": is_intern, "years_counted": score_years})
        if relevant:
            details["found_years"].append(score_years)

    # pattern 2: year ranges 2018-2022 or from 2018 to 2022
    for m in re.finditer(r"(20\d{2}|19\d{2})\s*(?:-|to)\s*(20\d{2}|19\d{2})", text):
        y1 = int(m.group(1)); y2=int(m.group(2))
        if y2 >= y1:
            dur = y2 - y1
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            context = text[start:end].lower()
            relevant = any(kw.lower() in context for kw in jd_keywords)
            is_intern = bool(re.search(r"\bintern(ship)?\b", context, flags=re.I))
            score_years = dur * (0.5 if is_intern else 1.0)
            if relevant:
                details["found_years"].append(score_years)
                details["raw_matches"].append({"match": m.group(0), "context": context[:120], "relevant": relevant, "intern": is_intern, "years_counted": score_years})

    relevant_years = sum(details["found_years"])
    # Cap and map to points
    capped = min(relevant_years, MAX_EXPERIENCE_YEARS_COUNTED)
    points = capped * EXPERIENCE_POINTS_PER_YEAR
    points = min(points, WEIGHTS["experience"])
    details["relevant_years_estimate"] = round(relevant_years, 2)
    details["capped_years"] = capped
    details["score"] = round(points, 2)
    details["max"] = WEIGHTS["experience"]

    return points, details

# ---------- SECTION-LEVEL DETAIL SCORE ----------

ACTION_VERBS = ["improved", "reduced", "increased", "built", "designed", "launched", "optimized", "created", "implemented", "developed"]

def section_detail_score(resume_text: str) -> Tuple[float, Dict]:
    """
    Heuristic to score the quality of job bullets:
      - count bullets (lines starting with '-' or '*')
      - reward lines containing numbers/percentages (metrics)
      - reward lines containing action verbs + tool mentions
    """
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    bullets = [l for l in lines if re.match(r"^[-\*\u2022]\s+", l) or len(l.split()) < 80 and (l.count('.')>=1 and len(l.split())>4)]
    score = 0.0
    max_pts = WEIGHTS["section"]
    metrics_found = 0
    action_found = 0
    tool_mentions = 0
    checked = 0

    for b in bullets:
        checked += 1
        has_metric = bool(re.search(r"\b\d+%|\b\d+\s*(?:days|weeks|months|years)|\b\d+\b", b))
        if has_metric:
            metrics_found += 1
            score += 1.5
        if any(av in b.lower() for av in ACTION_VERBS):
            action_found += 1
            score += 1.0
        # detect tools (simple heuristic: common tool words)
        if re.search(r"\b(python|java|c\+\+|c#|tensorflow|pytorch|django|flask|docker|kubernetes|aws|azure|gcp|sql|nosql)\b", b, flags=re.I):
            tool_mentions += 1
            score += 1.0

    # normalize by expected bullets
    expected_bullets = max(3, len(bullets))
    normalized = score / (expected_bullets * 3.5) * max_pts
    normalized = min(normalized, max_pts)
    details = {
        "bullets_counted": len(bullets),
        "metrics_found": metrics_found,
        "action_verbs_found": action_found,
        "tool_mentions": tool_mentions,
        "raw_score": round(score, 2),
        "normalized": round(normalized, 2),
        "max": max_pts
    }
    return normalized, details

# ---------- MICRO-LEVEL SKILLS / CERTS ----------

def micro_items_score(jd_keywords: List[str], resume_text: str) -> Tuple[float, Dict]:
    """
    Check for specific certifications, courses, and project mentions.
    Award small points per matched micro-item, cap at WEIGHTS['micro'].
    """
    resume_lower = resume_text.lower()
    matched = []
    missing = []
    # micro items are jd_keywords that are short/likely tools or certs
    candidates = [k for k in jd_keywords if len(k.split()) <= 3]
    # shortlist by likely tech/cert tokens
    micro_candidates = []
    for c in candidates:
        if re.search(r"[A-Za-z0-9\+\#\.\-]{2,}", c):
            micro_candidates.append(c)

    points_per_item = 1.5  # base
    raw_pts = 0.0
    for cand in micro_candidates:
        if cand.lower() in resume_lower or fuzz.token_set_ratio(cand.lower(), resume_lower) > 80:
            matched.append(cand)
            raw_pts += points_per_item
        else:
            missing.append(cand)

    raw_pts = min(raw_pts, WEIGHTS["micro"])
    details = {"matched": matched, "missing": missing, "raw": raw_pts, "max": WEIGHTS["micro"]}
    return raw_pts, details

# ---------- FINAL SCORING ----------

def aggregate_scores(components: Dict[str, Tuple[float, Dict]]) -> Dict:
    """
    components: {name: (score, details)}
    Returns aggregated report
    """
    total = 0.0
    breakdown = {}
    for name, (score, det) in components.items():
        total += score
        breakdown[name] = {"score": round(score, 2), "details": det}
    total = round(total, 2)
    # verdict
    if total >= HIGH_FIT_THRESHOLD:
        verdict = "High"
    elif total >= MEDIUM_FIT_THRESHOLD:
        verdict = "Medium"
    else:
        verdict = "Low"

    # find missing items from keyword/micro details
    missing = set()
    if "keyword" in breakdown:
        missing.update(breakdown["keyword"]["details"].get("missing_required", []))
    if "micro" in breakdown:
        missing.update(breakdown["micro"]["details"].get("missing", []))

    # suggestions
    suggestions = []
    if "keyword" in breakdown:
        km = breakdown["keyword"]["details"]
        if km.get("missing_required"):
            suggestions.append(f"Add missing must-have skills: {', '.join(km.get('missing_required'))}")
        if len(km.get("exact_matches", [])) < max(3, len(km.get("exact_matches", []))):
            suggestions.append("Add exact JD keywords in skills/work bullets (not just in a 'skills' blob).")
    if "section" in breakdown:
        sd = breakdown["section"]["details"]
        if sd.get("metrics_found", 0) < 2:
            suggestions.append("Quantify impact in work bullets (e.g., 'reduced X by 20%').")
        if sd.get("tool_mentions", 0) < 1:
            suggestions.append("Mention tools/technologies used for each role where applicable.")
    if "experience" in breakdown:
        ex = breakdown["experience"]["details"]
        if ex.get("relevant_years_estimate", 0) < 2:
            suggestions.append("Expand on relevant internships/projects or consider more domain work experience.")
    if len(suggestions) == 0:
        suggestions.append("Good alignment. Consider adding measurable outcomes for stronger scoring.")

    report = {
        "total_score": total,
        "breakdown": breakdown,
        "missing_items": list(missing),
        "verdict": verdict,
        "suggestions": suggestions
    }
    return report

# ---------- MAIN API ----------

def score_resume(jd_text: str, resume_path: str, openai_api_key: str = None) -> Dict:
    """
    Main function to call. Returns scoring report dict.
    """
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key must be provided either via openai_api_key or OPENAI_API_KEY env var.")

    # 1) Extract texts
    jd_text = clean_text(jd_text)
    resume_raw = extract_text(resume_path)
    resume_text = clean_text(resume_raw)

    # 2) Extract keywords from JD
    extracted = extract_candidate_keywords_from_jd(jd_text)
    jd_keywords = extracted["keywords"]
    jd_must = extracted["must_have"]

    # 3) Component scores
    k_score, k_det = keyword_and_fuzzy_score(jd_keywords, resume_text, must_have=jd_must)
    s_score, s_det = semantic_similarity_score(jd_text, resume_text, openai_api_key)
    e_score, e_det = extract_years_of_experience(resume_text, jd_keywords)
    sec_score, sec_det = section_detail_score(resume_text)
    m_score, m_det = micro_items_score(jd_keywords, resume_text)

    components = {
        "keyword": (k_score, k_det),
        "semantic": (s_score, s_det),
        "experience": (e_score, e_det),
        "section": (sec_score, sec_det),
        "micro": (m_score, m_det)
    }

    report = aggregate_scores(components)
    # attach some raw extras
    report["_meta"] = {
        "jd_keywords_sample": jd_keywords[:20],
        "jd_must_sample": jd_must,
        "resume_excerpt": resume_text[:1000]
    }
    return report

# ---------- CLI example ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score a resume against a job description.")
    parser.add_argument("--jd", required=True, help="Path to a job description text file (or pass text directly with --jd_text).")
    parser.add_argument("--jd_text", required=False, help="Pass JD text directly instead of a file.")
    parser.add_argument("--resume", required=True, help="Path to resume file (PDF/DOCX/TXT).")
    parser.add_argument("--openai_key", required=False, help="OpenAI API Key (or set OPENAI_API_KEY env var).")
    args = parser.parse_args()

    if args.jd_text:
        jd = args.jd_text
    else:
        with open(args.jd, "r", encoding="utf-8") as f:
            jd = f.read()

    result = score_resume(jd, args.resume, openai_api_key=args.openai_key)
    print(json.dumps(result, indent=2))
