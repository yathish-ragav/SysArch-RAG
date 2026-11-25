from pathlib import Path
from src.ingestion.chunker import (
    MAX_CHUNK_CHARS,
    _extract_frontmatter_and_body,
    _split_preserving_code_blocks,
    chunk_markdown_file,
)


def test_extract_frontmatter_title():
    raw = '---\ntitle: "Pod Disruption Budgets"\nweight: 10\n---\nBody text here.'
    title, body = _extract_frontmatter_and_body(raw)
    assert title == "Pod Disruption Budgets"
    assert body.strip() == "Body text here."


def test_split_preserves_code_fences_under_ceiling():
    yaml_block = "```yaml\n" + "\n".join(f"key_{i}: value_{i}" for i in range(90)) + "\n```"
    text = f"Intro paragraph.\n\n{yaml_block}\n\nOutro paragraph."
    chunks = _split_preserving_code_blocks(text, max_chars=MAX_CHUNK_CHARS)
    assert len(chunks) >= 2
    for ch in chunks:
        assert len(ch) <= MAX_CHUNK_CHARS
        assert ch.count("```") % 2 == 0


def test_chunk_markdown_file_metadata_and_deterministic_id(tmp_path: Path):
    doc = tmp_path / "kubernetes" / "concepts" / "pdb.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        '---\ntitle: "PDB Guide"\n---\n## Spec\nA PodDisruptionBudget sets `minAvailable` for voluntary evictions.\n',
        encoding="utf-8",
    )
    chunks_a = chunk_markdown_file(doc, tmp_path)
    chunks_b = chunk_markdown_file(doc, tmp_path)
    assert len(chunks_a) == 1
    assert chunks_a[0].chunk_id == chunks_b[0].chunk_id
    assert chunks_a[0].metadata.service == "kubernetes"
    assert chunks_a[0].metadata.version == "1.32"
    assert "PDB Guide > Spec" in chunks_a[0].metadata.section_hierarchy
