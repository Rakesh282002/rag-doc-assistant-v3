import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------

# Signals that indicate this is a resume / CV
_RESUME_SIGNALS = re.compile(
    r"\b(?:resume|curriculum\s+vitae|\bC\.?V\.?\b|"
    r"work\s+experience|professional\s+experience|professional\s+summary|"
    r"core\s+competencies|education|certifications?|technical\s+skills?)\b",
    re.IGNORECASE,
)

# Lines that are *only* a section header (handles all-caps or title-case)
# IMPORTANT: the entire alternation is inside (?:...) so ^ and $ anchor the whole match
_SECTION_HEADER = re.compile(
    r"^[ \t]*(?:"
    r"(?:PROFESSIONAL\s+)?(?:SUMMARY|PROFILE|OBJECTIVE|ABOUT\s+ME)|"
    r"CORE\s+COMPETENCIES|"
    r"(?:PROFESSIONAL\s+|WORK\s+)?EXPERIENCE|WORK\s+HISTORY|EMPLOYMENT(?:\s+HISTORY)?|"
    r"(?:TECHNICAL\s+|CORE\s+)?SKILLS?|COMPETENC(?:Y|IES)|"
    r"EDUCATION(?:AL\s+BACKGROUND)?|ACADEMIC(?:\s+BACKGROUND)?|"
    r"CERTIFICATIONS?|LICEN[CS]ES?|"
    r"ACHIEVEMENTS?|AWARDS?|ACCOMPLISHMENTS?|"
    r"PROJECTS?|PUBLICATIONS?|PATENTS?|LANGUAGES?|INTERESTS?|REFERENCES?"
    r")[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Job boundary: line that looks like "Company | Role" or "Company – Role (Year)"
# Also matches "Company\nRole\nDate" style via the date-line fallback
_JOB_BOUNDARY = re.compile(
    r"(?m)^"
    r"([A-Z][^\n]{2,70})"          # company / role line starting with uppercase
    r"(?:"
    r"\s*[|–—\-]\s*[^\n]{3,70}\n"  # separator then role on same line
    r"|"
    r"\n[^\n]{3,70}\n"              # OR role on next line
    r")"
    r"(?=[A-Z][a-z]|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})",
    re.UNICODE,
)

# Date range pattern (en-dash, em-dash, hyphen, or "to")
_DATE_RANGE = re.compile(
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|"
    r"\d{4})"
    r"[^\n]{0,30}"
    r"(?:[–—\-]|to)"
    r"[^\n]{0,30}"
    r"(?:Present|Current|Now|\d{4})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in {".docx", ".doc"}:
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return loader.load()


def split_documents(documents):
    full_text = "\n".join(d.page_content for d in documents)
    source = documents[0].metadata.get("source", "") if documents else ""

    if _is_resume(full_text):
        print("[CHUNKING] Resume detected — using section/role-based chunking")
        chunks = _split_resume(full_text, source)
        print(f"[CHUNKING] Produced {len(chunks)} chunks")
        return chunks

    print("[CHUNKING] Generic document — using RecursiveCharacterTextSplitter")
    return _split_generic(documents)


# ---------------------------------------------------------------------------
# Resume-specific chunking
# ---------------------------------------------------------------------------

def _is_resume(text: str) -> bool:
    return len(_RESUME_SIGNALS.findall(text[:3000])) >= 2


def _split_resume(text: str, source: str) -> list:
    chunks = []

    # Collect all section header positions
    boundaries = [(m.start(), m.group().strip()) for m in _SECTION_HEADER.finditer(text)]

    if not boundaries:
        # No recognised section headers — fall back to generic splitter
        return _split_generic([Document(page_content=text, metadata={"source": source})])

    # Content before first section header (name, contact, headline)
    pre = text[: boundaries[0][0]].strip()
    if pre:
        name = _extract_name(pre)
        chunks.append(
            Document(
                page_content=pre,
                metadata={"source": source, "section": "header", "name": name},
            )
        )

    # Process each section
    boundaries.append((len(text), "END"))
    for i, (start, header) in enumerate(boundaries[:-1]):
        end = boundaries[i + 1][0]
        body = text[start:end].strip()
        if len(body) < 40:
            continue

        section_type = _classify_section(header)

        if section_type == "experience":
            chunks.extend(_split_experience(body, source))
        elif section_type == "skills":
            skills_str = _extract_skills_list(body)
            chunks.append(
                Document(
                    page_content=body,
                    metadata={"source": source, "section": "skills", "skills": skills_str},
                )
            )
        else:
            chunks.append(
                Document(
                    page_content=body,
                    metadata={"source": source, "section": section_type, "header": header},
                )
            )

    # Secondary pass: split any chunk still larger than CHUNK_SIZE
    final_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    for chunk in chunks:
        if len(chunk.page_content) > CHUNK_SIZE:
            sub = splitter.split_documents([chunk])
            final_chunks.extend(sub)
        else:
            final_chunks.append(chunk)

    return final_chunks


def _split_experience(section_text: str, source: str) -> list:
    """One chunk per job role inside the experience section."""
    # Strip the section header line itself (e.g. "PROFESSIONAL EXPERIENCE")
    first_line, _, rest = section_text.partition("\n")
    if _SECTION_HEADER.match(first_line.strip()):
        section_text = rest.strip()
    # Find boundaries using the job-boundary pattern
    positions = [m.start() for m in _JOB_BOUNDARY.finditer(section_text)]

    # If we can't find distinct roles, keep section as-is
    if len(positions) < 2:
        return [
            Document(
                page_content=section_text,
                metadata={"source": source, "section": "experience"},
            )
        ]

    positions.append(len(section_text))
    role_docs = []

    # Any text before first detected role (section header line)
    prefix = section_text[: positions[0]].strip()
    if prefix:
        role_docs.append(
            Document(
                page_content=prefix,
                metadata={"source": source, "section": "experience_header"},
            )
        )

    for i, start in enumerate(positions[:-1]):
        end = positions[i + 1]
        role_text = section_text[start:end].strip()
        if len(role_text) < 30:
            continue
        meta = {"source": source, "section": "experience", **_extract_role_metadata(role_text)}
        role_docs.append(Document(page_content=role_text, metadata=meta))

    return role_docs


# ---------------------------------------------------------------------------
# Metadata extractors
# ---------------------------------------------------------------------------

def _extract_role_metadata(role_text: str) -> dict:
    meta = {}
    lines = [ln.strip() for ln in role_text.splitlines() if ln.strip()]
    if not lines:
        return meta

    first = lines[0]
    # "Company | Role" or "Company – Role" or "Company — Role"
    sep = re.match(r"^(.+?)\s*[|–—]\s*(.+)$", first)
    if sep:
        meta["company"] = sep.group(1).strip()
        meta["role"] = sep.group(2).strip()
    else:
        meta["company"] = first
        if len(lines) > 1:
            meta["role"] = lines[1]

    m = _DATE_RANGE.search(role_text)
    if m:
        meta["years"] = m.group().strip()

    return meta


def _extract_name(pre_text: str) -> str:
    lines = [ln.strip() for ln in pre_text.splitlines() if ln.strip()][:8]
    if not lines:
        return ""

    # Prefer a human-name looking line, skip obvious contact lines.
    contact_markers = {"@", "http", "www", "+", "linkedin", "github", "gmail"}
    for line in lines:
        lower = line.lower()
        if any(marker in lower for marker in contact_markers):
            continue
        if re.search(r"\d", line):
            continue
        if re.match(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}$", line):
            return line
        if re.match(r"^[A-Z]{2,}(?:\s[A-Z]{2,}){1,3}$", line):
            return line.title()

    # Fallback: first non-contact line.
    for line in lines:
        lower = line.lower()
        if not any(marker in lower for marker in contact_markers):
            return line

    return lines[0]


def _extract_skills_list(skills_text: str) -> str:
    skills = []
    for line in skills_text.splitlines()[1:]:          # skip section header
        for part in re.split(r"[•·,|/\t]", line):
            part = part.strip()
            if 2 <= len(part) <= 50 and not part.isnumeric():
                skills.append(part)
    return ", ".join(skills[:25])


def _classify_section(header: str) -> str:
    h = header.upper()
    if any(x in h for x in ["EXPERIENCE", "EMPLOYMENT", "WORK HIST"]):
        return "experience"
    if any(x in h for x in ["SKILL", "COMPETEN", "TECHNOLOG"]):
        return "skills"
    if "EDUCATION" in h or "ACADEMIC" in h:
        return "education"
    if "CERTIF" in h or "LICEN" in h:
        return "certifications"
    if any(x in h for x in ["SUMMARY", "PROFILE", "OBJECTIVE", "ABOUT"]):
        return "summary"
    if any(x in h for x in ["ACHIEVE", "AWARD", "ACCOMPLISH"]):
        return "achievements"
    if "PROJECT" in h:
        return "projects"
    return "other"


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------

def _split_generic(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)
