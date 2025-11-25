#!/usr/bin/env python3
"""
Obsidian Vault Enricher

Safe, idempotent utility to:
 - Back up the vault (markdown only) to _backup/YYYYmmdd_HHMMSS/
 - Ensure YAML frontmatter with title, tags, aliases, created, updated
 - Infer tags from paths, filenames, headings, and a small keyword map
 - Auto-link concepts across notes using [[Wiki Links]] while avoiding code blocks/frontmatter
 - Generate simple MOCs per directory, a root Index.md, and a Tag Index.md

This script avoids third-party deps and keeps edits conservative.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ------------------------- Helpers -------------------------


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
CODEBLOCK_FENCE_RE = re.compile(r"^\s*```")
WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^\)]+\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)")


KEYWORD_TAGS = {
    # Programming languages and stacks
    "python": ["python", "py"],
    "cpp": ["c++", "cpp"],
    "c": [" c "],
    "javascript": ["javascript", "js"],
    "django": ["django"],
    "flask": ["flask"],
    # CS topics
    "dsa": ["data structure", "algorithm", "dsa"],
    "dbms": ["dbms", "database", "b-tree", "b trees", "b-tree"],
    "system-design": ["system design", "sys design"],
    "networking": ["network", "ccna"],
    "os": ["operating system", "os "],
    "compiler": ["compiler"],
    # Math
    "calculus": ["calculus", "derivative", "integral", "limits"],
    "algebra": ["algebra", "linear algebra"],
    "geometry": ["geometry", "triangle", "circle", "ellipse", "parabola"],
    "probability": ["probability", "statistics", "random"],
    # ML & AI
    "ml": ["machine learning", "ml", "supervised", "unsupervised", "regression", "clustering"],
    "ai": ["ai", "agents", "rag"],
    # Robotics
    "ros": ["ros", "robotics", "realsense", "d435i"],
    # Career/Finance/Personal
    "finance": ["finance", "tax", "salary"],
    "career": ["career", "work", "resume", "aptiv"],
    "hobby": ["chess", "badminton", "trip", "books", "ikigai", "think fast"],
    # Security
    "security": ["security", "cyber", "cybersec"],
}


EXCLUDE_DIRS = {
    ".obsidian",
    ".git",
    ".trash",
    "alltagfile",
    "_backup",
    ".smart-env",
}


def now_iso() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def sanitize_to_words(text: str) -> List[str]:
    # Remove emojis and punctuation-ish, split to words, lowercase
    cleaned = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    words = [w.lower() for w in cleaned.split() if w]
    return words


def slugify(text: str) -> str:
    return "-".join(sanitize_to_words(text))


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_text(path: Path, data: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(data)


def ensure_backup(vault: Path) -> Path:
    backup_root = vault / "_backup"
    backup_root.mkdir(exist_ok=True)
    ts_dir = backup_root / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_dir.mkdir()
    for p in vault.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        rel = p.relative_to(vault)
        dst = ts_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    return ts_dir


def split_frontmatter_and_body(text: str) -> Tuple[Dict[str, object], str]:
    m = FRONTMATTER_RE.match(text)
    fm: Dict[str, object] = {}
    body = text
    if m:
        fm_text = m.group(1)
        body = text[m.end() :]
        # Very small YAML-ish parser for our keys
        current_key: Optional[str] = None
        for line in fm_text.splitlines():
            if not line.strip():
                continue
            if re.match(r"^[A-Za-z0-9_\-]+:\s*", line):
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if val == "":
                    fm[key] = []
                    current_key = key
                else:
                    fm[key] = val
                    current_key = None
            elif line.strip().startswith("- ") and current_key:
                fm.setdefault(current_key, [])
                assert isinstance(fm[current_key], list)
                fm[current_key].append(line.strip()[2:].strip())
            else:
                # ignore unknown forms to keep this simple
                pass
    return fm, body


def dump_frontmatter(fm: Dict[str, object]) -> str:
    lines: List[str] = ["---"]
    # Emit in stable order
    order = ["title", "aliases", "tags", "created", "updated"]
    for key in order:
        if key not in fm or fm[key] in (None, [], ""):
            continue
        val = fm[key]
        if isinstance(val, list):
            lines.append(f"{key}:")
            for item in val:
                lines.append(f"- {item}")
        else:
            lines.append(f"{key}: {val}")
    lines.append("---\n")
    return "\n".join(lines)


def determine_title(path: Path, body: str, existing_title: Optional[str]) -> str:
    if existing_title and str(existing_title).strip():
        return str(existing_title).strip()
    # Try first heading
    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if m:
            return m.group(1).strip()
        if line.strip():
            break
    # Fallback to filename
    return path.stem


def infer_tags(path: Path, title: str, body: str) -> Set[str]:
    tags: Set[str] = set()
    # From path components
    for part in path.parts[:-1]:
        if part.startswith('.'):
            continue
        for w in sanitize_to_words(part):
            if len(w) > 2:
                tags.add(w)
    # From title
    for w in sanitize_to_words(title):
        if len(w) > 2:
            tags.add(w)
    # From keyword map
    lowered = f" {title.lower()} \n {body[:2000].lower()} "  # limit body slice for speed
    for tag, needles in KEYWORD_TAGS.items():
        for needle in needles:
            if needle in lowered:
                tags.add(tag)
                break
    # Collapse some common words
    common_stop = {"and", "the", "for", "with", "from", "into", "about", "your", "note", "notes"}
    tags = {t for t in tags if t not in common_stop}
    # Filter noisy numeric tokens (years, counts)
    tags = {t for t in tags if not t.isdigit() and not re.fullmatch(r"\d{2,4}", t)}
    # Namespaces for clarity
    namespaced: Set[str] = set()
    for t in tags:
        if t in {"python", "cpp", "javascript", "django", "flask", "ros"}:
            namespaced.add(f"lang/{t}")
        elif t in {"ml", "ai", "dsa", "dbms", "system-design", "networking", "compiler", "os"}:
            namespaced.add(f"cs/{t}")
        elif t in {"calculus", "algebra", "geometry", "probability"}:
            # Use the user's preferred mathematics/ namespace
            namespaced.add(f"mathematics/{t}")
        elif t in {"finance"}:
            namespaced.add(f"life/{t}")
        elif t in {"career", "hobby", "security"}:
            namespaced.add(f"misc/{t}")
        else:
            namespaced.add(t)
    return namespaced


def compute_aliases(path: Path, title: str, body: str, existing_aliases: Iterable[str]) -> List[str]:
    aliases: List[str] = []
    seen: Set[str] = set()
    def add(s: str):
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            aliases.append(s)

    for a in existing_aliases or []:
        add(a)
    add(path.stem.replace("_", " ").replace("-", " "))
    if title != path.stem:
        add(title)
    # Shortened versions
    words = title.split()
    if len(words) > 3:
        add(" ".join(words[:3]))
    return aliases


def protect_regions(lines: List[str]) -> List[Tuple[str, bool]]:
    """Return (line, linkable) pairs where code blocks and frontmatter are not linkable."""
    protected: List[Tuple[str, bool]] = []
    in_code = False
    in_frontmatter = False
    fm_seen = 0
    for idx, line in enumerate(lines):
        if idx == 0 and line.strip() == "---":
            in_frontmatter = True
            fm_seen = 1
            protected.append((line, False))
            continue
        if in_frontmatter:
            protected.append((line, False))
            if line.strip() == "---":
                in_frontmatter = False
            continue
        if CODEBLOCK_FENCE_RE.match(line):
            in_code = not in_code
            protected.append((line, False))
            continue
        protected.append((line, not in_code))
    return protected


def build_title_index(vault: Path) -> Dict[str, str]:
    """Map normalized names -> canonical title."""
    index: Dict[str, str] = {}
    for p in vault.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        text = read_text(p)
        fm, body = split_frontmatter_and_body(text)
        title = determine_title(p, body, fm.get("title") if isinstance(fm.get("title"), str) else None)
        names = {title, p.stem.replace("_", " ").replace("-", " ")}
        # Also include simple lowercase tokens joined
        for n in list(names):
            index[n.lower()] = title
        # Include aliases
        if isinstance(fm.get("aliases"), list):
            for a in fm["aliases"]:
                index[str(a).lower()] = title
    return index


def linkify_text(body: str, self_title: str, title_index: Dict[str, str]) -> str:
    # Prepare sorted keys: longer phrases first to reduce partial matches
    candidates = [(k, v) for k, v in title_index.items() if v != self_title]
    candidates.sort(key=lambda kv: len(kv[0]), reverse=True)

    def replace_in_line(line: str) -> str:
        if not line.strip():
            return line
        # Skip if already contains links heavily
        if WIKILINK_RE.search(line) or MD_LINK_RE.search(line):
            return line
        for key, target_title in candidates:
            # Word-boundary match, case-insensitive
            pattern = re.compile(rf"(?<!\[\[)(?<!\w)({re.escape(key)})(?!\w)(?![^\[]*\]\])", re.IGNORECASE)
            # Limit 2 replacements per target per line to avoid overlinking
            line_new, n = pattern.subn(rf"[[{target_title}]]", line, count=2)
            line = line_new
        return line

    lines = body.splitlines()
    protected = protect_regions(lines)
    out_lines: List[str] = []
    for line, linkable in protected:
        out_lines.append(replace_in_line(line) if linkable else line)
    return "\n".join(out_lines)


def ensure_frontmatter(path: Path, text: str) -> str:
    fm, body = split_frontmatter_and_body(text)
    title = determine_title(path, body, fm.get("title") if isinstance(fm.get("title"), str) else None)
    existing_aliases: List[str] = []
    if isinstance(fm.get("aliases"), list):
        existing_aliases = [str(x) for x in fm["aliases"]]
    tags_existing: Set[str] = set()
    if isinstance(fm.get("tags"), list):
        tags_existing = {str(x) for x in fm["tags"]}

    tags = infer_tags(path, title, body)
    tags |= tags_existing

    aliases = compute_aliases(path, title, body, existing_aliases)

    # created/updated
    created = fm.get("created") if isinstance(fm.get("created"), str) else None
    if not created:
        try:
            stat = path.stat()
            created = dt.datetime.fromtimestamp(stat.st_mtime).astimezone().replace(microsecond=0).isoformat()
        except Exception:
            created = now_iso()

    updated = now_iso()

    new_fm = {
        "title": title,
        "aliases": aliases,
        "tags": sorted(tags),
        "created": created,
        "updated": updated,
    }

    header = dump_frontmatter(new_fm)
    return header + body.lstrip("\n")


def generate_moc_for_dir(dir_path: Path, vault: Path) -> None:
    if any(part in EXCLUDE_DIRS for part in dir_path.parts):
        return
    items: List[str] = []
    for p in sorted(dir_path.glob("*.md")):
        if p.name.startswith("MOC - "):
            continue
        text = read_text(p)
        fm, body = split_frontmatter_and_body(text)
        title = determine_title(p, body, fm.get("title") if isinstance(fm.get("title"), str) else None)
        items.append(f"- [[{title}]]")
    for sub in sorted([d for d in dir_path.iterdir() if d.is_dir()]):
        if any(part in EXCLUDE_DIRS for part in sub.parts):
            continue
        md_candidates = list(sub.glob("*.md"))
        if md_candidates:
            items.append(f"- [[MOC - {sub.name}]]")
    if not items:
        return
    moc_name = f"MOC - {dir_path.name}.md"
    moc_path = dir_path / moc_name
    content = ["# Map of Content", "", *items, ""]
    write_text(moc_path, "\n".join(content))


def generate_root_index(vault: Path) -> None:
    sections: List[str] = ["# Index", ""]
    for top in sorted([d for d in vault.iterdir() if d.is_dir()]):
        if top.name in EXCLUDE_DIRS:
            continue
        sections.append(f"## {top.name}")
        moc = top / f"MOC - {top.name}.md"
        if moc.exists():
            sections.append(f"- [[MOC - {top.name}]]")
        for p in sorted(top.glob("*.md"))[:20]:  # cap to avoid bloat
            text = read_text(p)
            fm, body = split_frontmatter_and_body(text)
            title = determine_title(p, body, fm.get("title") if isinstance(fm.get("title"), str) else None)
            sections.append(f"- [[{title}]]")
        sections.append("")
    write_text(vault / "Index.md", "\n".join(sections))


def generate_tag_index(vault: Path, all_tags: Set[str]) -> None:
    lines = ["# Tag Index", "", "Search by clicking tags below:", ""]
    for t in sorted(all_tags):
        lines.append(f"- #{t}")
    lines.append("")
    write_text(vault / "Tag Index.md", "\n".join(lines))


def process_vault(vault: Path, dry_run: bool = False) -> None:
    if not vault.exists():
        print(f"Vault does not exist: {vault}", file=sys.stderr)
        sys.exit(2)

    print("Creating backup...", flush=True)
    backup_dir = ensure_backup(vault)
    print(f"Backup at: {backup_dir}")

    print("Indexing titles...", flush=True)
    title_index = build_title_index(vault)

    all_tags: Set[str] = set()
    changed_files = 0

    print("Updating notes...", flush=True)
    for path in vault.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        original = read_text(path)
        updated_fm_text = ensure_frontmatter(path, original)
        fm, body_after_fm = split_frontmatter_and_body(updated_fm_text)
        # Linkify using updated index
        title = str(fm.get("title")) if fm.get("title") else path.stem
        linked_body = linkify_text(body_after_fm, title, title_index)
        final_text = dump_frontmatter(fm) + linked_body

        # Collect tags
        if isinstance(fm.get("tags"), list):
            all_tags.update([str(x) for x in fm["tags"]])

        if final_text != original:
            changed_files += 1
            if not dry_run:
                write_text(path, final_text)

    print(f"Notes updated: {changed_files}")

    print("Generating MOCs...", flush=True)
    # Per directory MOC
    for d in sorted([p for p in vault.iterdir() if p.is_dir()]):
        if d.name in EXCLUDE_DIRS:
            continue
        for sub in [d] + [p for p in d.rglob("*") if p.is_dir() and not any(part in EXCLUDE_DIRS for part in p.parts)]:
            generate_moc_for_dir(sub, vault)

    print("Generating Index and Tag Index...", flush=True)
    generate_root_index(vault)
    generate_tag_index(vault, all_tags)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Obsidian vault")
    parser.add_argument("--vault", type=str, required=True, help="Absolute path to vault root")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = parser.parse_args()

    vault = Path(args.vault).resolve()
    process_vault(vault, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
