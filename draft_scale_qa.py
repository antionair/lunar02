import csv
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data/scale/qa_source_candidates.tsv"
OUTPUT = ROOT / "data/scale/qa_draft.jsonl"
REPORT = ROOT / "data/scale/qa_machine_check.json"

NAME = r"(?:[A-Z][A-Za-z'’-]+|van|de|der|of|the|and|di|du|la|le){1,7}"
PERSON = r"(?:[A-Z][A-Za-z'’-]+|van|de|der|di|du|la|le){1,5}"
PLACE = r"[^.,;]{2,80}"
QUOTAS = {"who": 80, "what": 30, "which": 15, "how-many": 15, "when": 50, "where": 30}
GENERIC_NAMES = {"the", "and", "a", "an", "he", "she", "it", "they", "museum", "laboratory", "book", "company", "university", "college", "school", "center", "centre", "project", "program", "first", "second", "one", "two", "area", "service", "castle", "palace", "river", "bear", "as", "this", "that", "city", "street", "process", "types", "cemetery", "park", "estate", "newly", "actually", "foundation", "images", "what", "who", "where", "when", "which", "how", "plant", "page", "society", "reservation", "standards"}
PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "who", "what", "where", "when", "which", "how"}
BAD_ANSWERS = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "who", "what", "where", "when", "which", "how", "he", "she", "it", "they", "man", "woman", "person", "people", "george", "henry", "thomas", "frank", "richard", "hal", "joel", "diana", "rick", "rocky", "western"}
SINGLE_NAME_BLOCK = {"make", "bodies", "west", "stories", "estrogens", "project", "museum", "laboratory", "company", "school", "center", "centre"}
LOCATION_BAD = {"is", "was", "were", "has", "had", "became", "spent", "located", "built", "served", "published", "died", "born", "founded", "established", "two", "one", "original"}

PATTERNS = [
    ("who", "written_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?written by (?P<answer>{PERSON})"), "Who wrote {name}?"),
    ("who", "author_of", re.compile(rf"\b(?P<answer>{PERSON}) is the author of (?P<name>{NAME})"), "Who wrote {name}?"),
    ("who", "invented_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?invented by (?P<answer>{PERSON})"), "Who invented {name}?"),
    ("who", "invented", re.compile(rf"\b(?P<answer>{PERSON}) invented (?P<name>{NAME})"), "Who invented {name}?"),
    ("who", "discovered_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?discovered by (?P<answer>{PERSON})"), "Who discovered {name}?"),
    ("who", "founded_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?founded by (?P<answer>{PERSON})"), "Who founded {name}?"),
    ("who", "president", re.compile(rf"\bpresident of (?P<name>{NAME}) is (?P<answer>{PERSON})"), "Who is the president of {name}?"),
    ("who", "created_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?created by (?P<answer>{PERSON})"), "Who created {name}?"),
    ("who", "developed_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?developed by (?P<answer>{PERSON})"), "Who developed {name}?"),
    ("who", "designed_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?designed by (?P<answer>{PERSON})"), "Who designed {name}?"),
    ("who", "built_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?built by (?P<answer>{PERSON})"), "Who built {name}?"),
    ("who", "composed_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?composed by (?P<answer>{PERSON})"), "Who composed {name}?"),
    ("who", "directed_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?directed by (?P<answer>{PERSON})"), "Who directed {name}?"),
    ("who", "painted_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?painted by (?P<answer>{PERSON})"), "Who painted {name}?"),
    ("who", "produced_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?produced by (?P<answer>{PERSON})"), "Who produced {name}?"),
    ("who", "made_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?made by (?P<answer>{PERSON})"), "Who made {name}?"),
    ("who", "established_by", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?established by (?P<answer>{PERSON})"), "Who established {name}?"),
    ("what", "capital", re.compile(rf"\bcapital of (?P<name>{NAME}) is (?P<answer>{PERSON})"), "What is the capital of {name}?"),
    ("what", "language", re.compile(rf"\bofficial language of (?P<name>{NAME}) is (?P<answer>{PERSON})"), "What is the official language of {name}?"),
    ("what", "currency", re.compile(rf"\bcurrency of (?P<name>{NAME}) is (?P<answer>{PERSON})"), "What is the currency of {name}?"),
    ("what", "nickname", re.compile(rf"\bnickname of (?P<name>{NAME}) is (?P<answer>{PERSON})"), "What is the nickname of {name}?"),
    ("what", "motto", re.compile(rf"\bmotto of (?P<name>{NAME}) is (?P<answer>[^.\"']{2,60})"), "What is the motto of {name}?"),
    ("what", "official_name", re.compile(rf"\bofficial name of (?P<name>{NAME}) is (?P<answer>{PERSON})"), "What is the official name of {name}?"),
    ("what", "category", re.compile(rf"\b(?P<name>{NAME}) is a (?P<answer>[a-z][a-z -]{2,40})"), "What is {name}?"),
    ("what", "capital_reverse", re.compile(rf"\b(?P<answer>{NAME}) (?:was |is )?the capital of (?P<name>{NAME})"), "What was the capital of {name}?"),
    ("what", "capital_was", re.compile(rf"\bcapital of (?P<name>{NAME}) (?:was |is )(?P<answer>{PERSON})"), "What is the capital of {name}?"),
    ("what", "known_as", re.compile(rf"\b(?P<name>{NAME}) is (?:also )?(?:known|called) as (?P<answer>[^.\"']{2,60})"), "What is {name} also known as?"),
    ("which", "capital", re.compile(rf"\bcapital of (?P<name>{NAME}) is (?P<answer>{PERSON})"), "Which city is the capital of {name}?"),
    ("which", "capital_reverse", re.compile(rf"\b(?P<name>{NAME}) is the capital of (?P<answer>{NAME})"), "Which place is {name} the capital of?"),
    ("which", "capital_was_reverse", re.compile(rf"\b(?P<name>{NAME}) was the capital of (?P<answer>{NAME})"), "Which place was {name} the capital of?"),
    ("which", "located", re.compile(rf"\b(?P<name>{NAME}) is located in (?P<answer>{PLACE})"), "Which place is {name} located in?"),
    ("how-many", "population_of", re.compile(rf"\bpopulation of (?P<name>{NAME}) (?:is|was) (?P<answer>[\d,]+)"), "How many people live in {name}?"),
    ("how-many", "population_has", re.compile(rf"\b(?P<name>{NAME}) has a population of (?P<answer>[\d,]+)"), "How many people live in {name}?"),
    ("how-many", "people_live", re.compile(rf"\b(?P<answer>[\d,]+) people live in (?P<name>{NAME})"), "How many people live in {name}?"),
    ("how-many", "students", re.compile(rf"\b(?P<name>{NAME}) has (?P<answer>[\d,]+) students"), "How many students does {name} have?"),
    ("how-many", "employees", re.compile(rf"\b(?P<name>{NAME}) has (?P<answer>[\d,]+) employees"), "How many employees does {name} have?"),
    ("how-many", "members", re.compile(rf"\b(?P<name>{NAME}) has (?P<answer>[\d,]+) members"), "How many members does {name} have?"),
    ("how-many", "area", re.compile(rf"\b(?P<name>{NAME}) covers an area of (?P<answer>[\d,]+) square"), "How many square kilometers does {name} cover?"),
    ("how-many", "population_had", re.compile(rf"\b(?P<name>{NAME}) had a population of (?P<answer>[\d,]+)"), "How many people lived in {name}?"),
    ("how-many", "population_was", re.compile(rf"\b(?P<name>{NAME})(?:'s|’) population was (?P<answer>[\d,]+)"), "How many people lived in {name}?"),
    ("how-many", "population_approximately", re.compile(rf"\b(?P<name>{NAME}) had a population of approximately (?P<answer>[\d,]+)"), "How many people lived in {name}?"),
    ("how-many", "enrollment", re.compile(rf"\benrollment (?:at|of) (?P<name>{NAME}) (?:is|was) (?P<answer>[\d,]+)"), "How many students are enrolled at {name}?"),
    ("when", "born", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?born (?:on |in )(?P<answer>(?:[A-Z][a-z]+ \d{{1,2}}, )?\d{{4}})"), "When was {name} born?"),
    ("when", "died", re.compile(rf"\b(?P<name>{NAME}) (?:died|passed away) (?:on |in )(?P<answer>(?:[A-Z][a-z]+ \d{{1,2}}, )?\d{{4}})"), "When did {name} die?"),
    ("when", "founded", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?founded (?:in |on )(?P<answer>\d{{4}})"), "When was {name} founded?"),
    ("when", "established", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?established (?:in |on )(?P<answer>\d{{4}})"), "When was {name} established?"),
    ("when", "published", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?published in (?P<answer>\d{{4}})"), "When was {name} published?"),
    ("where", "located", re.compile(rf"\b(?P<name>{NAME}) is located (?:in|on) (?P<answer>{PLACE})"), "Where is {name} located?"),
    ("where", "headquartered", re.compile(rf"\b(?P<name>{NAME}) is headquartered (?:in|at) (?P<answer>{PLACE})"), "Where is {name} headquartered?"),
    ("where", "birthplace", re.compile(rf"\b(?P<name>{NAME}) (?:was |is )?born in (?P<answer>{PLACE})"), "Where was {name} born?"),
]


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def complete_excerpt(text):
    text = text[:6000].replace("\r", " ").replace("\n", " ")
    positions = [text.rfind(mark) for mark in [". ", "! ", "? "]]
    end = max(positions)
    if end < 80:
        return None
    return text[:end + 1].strip()


def expand_before(text, name, event):
    particle = r"(?:[A-Z][A-Za-z'’-]+|van|de|der|di|du|la|le|von)"
    for match in re.finditer(rf"\b((?:{particle} ){{1,5}}{particle})\s+(?:was |is )?{event}\b", text):
        candidate = " ".join(match.group(1).split())
        if candidate.split()[-1].lower() == name.split()[-1].lower():
            return candidate
    return name


def expand_after_by(text, name):
    particle = r"(?:[A-Z][A-Za-z'’-]+|van|de|der|di|du|la|le|von)"
    for match in re.finditer(rf"\bby\s+((?:{particle} ){{1,5}}{particle})\b", text):
        candidate = " ".join(match.group(1).split())
        if candidate.split()[-1].lower() == name.split()[-1].lower():
            return candidate
    return name


def clean_name(name):
    name = re.sub(r"^(?:When|Where|Who|What|Which|How)\s+", "", name)
    name = re.sub(r"^(?:Presidency|Career|State|Life|Early life|View Full Biography|General|Sir|Dr|Professor)\s+", "", name)
    return " ".join(name.split()).strip(" ,;:")


def shorten_answer(kind, answer):
    answer = " ".join(answer.split()).rstrip(" ,;:")
    if kind in {"when"}:
        match = re.search(r"((?:[A-Z][a-z]+ \d{1,2}, )?\d{4})$", answer)
        return match.group(1) if match else None
    if kind == "how-many":
        match = re.search(r"([\d,]+)$", answer)
        return match.group(1) if match else None
    if kind in {"where"}:
        answer = re.split(r",|;| and | as well as | to ", answer)[0].strip()
    answer = re.sub(r"^(?:the )?(?:town|city|state|region|heart of the region) of ", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"^the ", "", answer, flags=re.IGNORECASE)
    words = answer.split()
    limit = 5 if kind in {"who", "what", "which"} else 10
    if len(words) > limit:
        return None
    return answer


def valid(kind, name, answer, excerpt=""):
    if not name or len(name) < 2 or name.lower() in GENERIC_NAMES or re.fullmatch(r"[IVXLC]+", name):
        return False
    if not answer or answer.lower() in PRONOUNS or answer.lower() in BAD_ANSWERS or not answer.isascii() or not name.isascii():
        return False
    if kind == "who" and len(name.split()) < 2:
        if name.lower() in SINGLE_NAME_BLOCK or excerpt.lower().count(name.lower()) < 1:
            return False
    words = answer.split()
    if answer.lower().startswith(("north of", "south of", "east of", "west of", "heart of", "center of")):
        return False
    if kind == "when":
        return bool(re.fullmatch(r"(?:[A-Z][a-z]+ \d{1,2}, )?\d{4}", answer))
    if kind == "how-many":
        return bool(re.fullmatch(r"[\d,]+", answer))
    if kind == "who":
        particles = {"van", "de", "der", "di", "du", "la", "le", "von"}
        return 1 <= len(words) <= 5 and not re.search(r"\d", answer) and all(word[:1].isupper() or word.lower() in particles for word in words)
    if kind in {"what", "which"}:
        return 1 <= len(words) <= 5 and not re.search(r"\d", answer)
    if kind == "where":
        lowered = {word.strip(".,;:()").lower() for word in words}
        if answer.lower().startswith(("north of", "south of", "east of", "west of", "heart of", "center of")):
            return False
        return 2 <= len(words) <= 10 and not re.search(r"\d", answer) and not (lowered & LOCATION_BAD)
    return False


def aliases(answer):
    values = [answer]
    if answer.lower().startswith("the "):
        values.append(answer[4:])
    return list(dict.fromkeys(values))


def main():
    counts = {key: 0 for key in QUOTAS}
    rows = []
    seen = set()
    failures = []
    with INPUT.open(encoding="utf-8") as f:
        for source in csv.DictReader(f, delimiter="\t"):
            excerpt = complete_excerpt(source["text"])
            if excerpt is None:
                failures.append({"document_id": source["document_id"], "reason": "excerpt_has_no_complete_sentence"})
                continue
            candidates = []
            for kind, pattern_name, pattern, template in PATTERNS:
                match = pattern.search(excerpt)
                if not match:
                    continue
                name = " ".join(match.group("name").split()).strip(" ,;:")
                answer = " ".join(match.group("answer").split()).strip(" ,;:")
                if pattern_name in {"born", "died", "birthplace"}:
                    name = clean_name(expand_before(excerpt, name, "born" if pattern_name != "died" else "died"))
                    if len(name.split()) < 2:
                        continue
                else:
                    name = clean_name(name)
                if kind == "who":
                    by_patterns = {"written_by", "invented_by", "discovered_by", "founded_by", "created_by", "developed_by", "designed_by", "built_by", "composed_by", "directed_by", "painted_by"}
                    answer = expand_after_by(excerpt, answer) if pattern_name in by_patterns else expand_before(excerpt, answer, "invented" if pattern_name == "invented" else "the author of")
                answer = shorten_answer(kind, answer)
                if answer is None or not valid(kind, name, answer, excerpt):
                    failures.append({"document_id": source["document_id"], "pattern": pattern_name, "reason": "semantic_validation_failed", "name": name, "answer": answer})
                    continue
                question = template.format(name=name, answer=answer)
                candidates.append((kind, pattern_name, question, answer))
            for kind, pattern_name, question, answer in candidates:
                if counts[kind] >= QUOTAS[kind] or question in seen:
                    continue
                seen.add(question)
                counts[kind] += 1
                rows.append({
                    "id": f"qa-{len(rows) + 1:03d}",
                    "question": question,
                    "answer": answer,
                    "accepted_aliases": aliases(answer),
                    "question_type": kind,
                    "source_document_id": source["document_id"],
                    "source_domain": source["domain"],
                    "source_token_start": int(source["token_start"]),
                    "source_token_end": int(source["token_end"]),
                    "source_excerpt": excerpt,
                    "status": "draft_requires_user_audit",
                })
                break
            if len(rows) >= 200:
                break
    when_where = counts["when"] + counts["where"]
    checks = {
        "total": len(rows),
        "type_counts": counts,
        "when_where_count": when_where,
        "when_where_ratio": when_where / max(1, len(rows)),
        "when_where_within_60_percent": when_where <= 0.6 * len(rows),
        "all_questions_have_allowed_type": all(row["question_type"] in QUOTAS for row in rows),
        "all_where_answers_location_shaped": all(valid("where", "Example", row["answer"]) for row in rows if row["question_type"] == "where"),
        "all_excerpts_end_at_sentence_boundary": all(row["source_excerpt"].rstrip().endswith((".", "!", "?")) for row in rows),
        "all_have_accepted_aliases": all(bool(row["accepted_aliases"]) for row in rows),
        "failure_sample": failures[:50],
    }
    report = {"pass": len(rows) == 200 and checks["when_where_within_60_percent"] and checks["all_where_answers_location_shaped"] and checks["all_excerpts_end_at_sentence_boundary"] and checks["all_have_accepted_aliases"], **checks}
    if not report["pass"]:
        raise RuntimeError(json.dumps(report, indent=2))
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
