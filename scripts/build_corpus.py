"""Собирает JSONL-корпус из закреплённого снимка Markdown."""

import argparse
import json
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
SHORTCODE = re.compile(r"\{\{(?:<\s*(.*?)\s*>|%\s*(.*?)\s*%)\}\}", re.DOTALL)
HUGO_HEADING = {"prerequisites": "Предварительные требования", "whatsnext": "Что дальше"}


def clean_markup(markdown):
    """Убирает служебную разметку Hugo, сохраняя номера строк."""
    def replace_shortcode(match):
        shortcode = match[1] or match[2]
        heading = re.fullmatch(r'heading "([^"]+)"', shortcode)
        label = re.search(r'\btext="([^"]+)"', shortcode)
        term = re.search(r'\bterm_id="([^"]+)"', shortcode)
        if heading:
            replacement = HUGO_HEADING.get(heading[1], heading[1])
        elif shortcode.startswith("glossary_tooltip ") and (label or term):
            replacement = (label or term)[1]
        else:
            replacement = ""
        return replacement + match[0].count("\n") * "\n"

    without_comments = COMMENT.sub(lambda match: match[0].count("\n") * "\n", markdown)
    return SHORTCODE.sub(replace_shortcode, without_comments)


def markdown_sections(lines, first_line, title):
    """Выделяет разделы ATX, оставляя заголовки внутри примеров в тексте."""
    headings = [(0, title)]
    heading_line = first_line
    blocks = []
    block = []
    fence = None
    for line_number, line in enumerate(lines, start=first_line):
        marker = FENCE.match(line)
        if fence:
            block.append((line_number, line))
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = None
                blocks.append(block)
                block = []
            continue

        heading = HEADING.match(line)
        if marker or heading or not line.strip():
            if block:
                blocks.append(block)
                block = []
        if marker:
            fence = marker[1]
            block.append((line_number, line))
        elif heading:
            if blocks:
                yield heading_line, [name for _, name in headings], blocks
            level = len(heading[1])
            name = re.sub(r"\s+\{#[^}]+\}$", "", heading[2])
            name = re.sub(r"\s+#+$", "", name)
            headings = [(depth, label) for depth, label in headings if depth < level]
            headings.append((level, name))
            heading_line = line_number
            blocks = []
        elif line.strip():
            block.append((line_number, line))

    if block:
        blocks.append(block)
    if blocks:
        yield heading_line, [name for _, name in headings], blocks


def pack_blocks(blocks, max_chars):
    """Группирует абзацы в рамках раздела, сохраняя блоки кода целиком."""
    text = ""
    start = end = None
    for block in blocks:
        body = "\n".join(line for _, line in block)
        if text and len(text) + 2 + len(body) > max_chars:
            yield start, end, text
            text = ""
        if not text:
            start = block[0][0]
        text = f"{text}\n\n{body}" if text else body
        end = block[-1][0]
    if text:
        yield start, end, text


def document_chunks(markdown, document, manifest, max_chars):
    lines = markdown.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"Нет YAML-шапки: {document['path']}")
    frontmatter_end = lines.index("---", 1)
    frontmatter = lines[1:frontmatter_end]
    title_line = next(line for line in frontmatter if line.startswith("title:"))
    title = title_line.removeprefix("title:").strip().strip('"')
    first_line = frontmatter_end + 2
    content = clean_markup("\n".join(lines[frontmatter_end + 1:])).split("\n")
    version = manifest["commit"]
    source = f"{manifest['repository']}/blob/{version}/{document['path']}"
    for heading_line, headings, blocks in markdown_sections(content, first_line, title):
        section_id = f"{document['id']}:{heading_line}"
        for start, end, text in pack_blocks(blocks, max_chars):
            url = f"{source}#L{start}-L{end}"
            yield {
                "chunk_id": str(uuid5(NAMESPACE_URL, url)),
                "section_id": section_id,
                "document_id": document["id"],
                "version": version,
                "heading": headings[-1],
                "heading_path": headings,
                "source_url": url,
                "line_start": start,
                "line_end": end,
                "text": text,
            }


def build_corpus(root, max_chars=1400):
    manifest = json.loads((root / "data/sources.json").read_text(encoding="utf-8"))
    chunks = []
    for document in manifest["documents"]:
        source = root / "data/raw" / f"{document['id']}.md"
        markdown = source.read_text(encoding="utf-8")
        chunks.extend(document_chunks(markdown, document, manifest, max_chars))
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/chunks.jsonl")
    args = parser.parse_args()
    chunks = build_corpus(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Documents: {len({chunk['document_id'] for chunk in chunks})}. Chunks: {len(chunks)}.")


if __name__ == "__main__":
    main()
