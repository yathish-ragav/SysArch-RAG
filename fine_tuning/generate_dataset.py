
import json
from pathlib import Path
import random
import re

from src.ingestion.chunker import chunk_markdown_file


RAW_DIR = Path("data/raw_docs")
OUT_DIR = Path("fine_tuning")

SYSTEM_PROMPT = (
    "You are a strict technical retrieval grader for Kubernetes, AWS EKS, and etcd documentation. "
    "Evaluate whether <context> contains enough verified facts to answer <question>. "
    'Respond ONLY with valid JSON: {"is_sufficient": bool, "confidence": float, "reasoning": str}.'
)


def _question_from_chunk(
    service: str,
    hierarchy: str,
    content: str,
) -> str:

    parts = [
        p.strip()
        for p in hierarchy.split(">")
        if p.strip() and "whatsnext" not in p.lower()
    ]

    topic = parts[-1] if parts else service

    flags = re.findall(
        r"--[a-zA-Z0-9_-]{3,}",
        content,
    )

    kinds = re.findall(
        r"\b[A-Z][a-zA-Z0-9]{4,}\b",
        content,
    )

    if flags:
        return (
            f"How do you configure `{flags}` "
            f"for {topic} in {service}?"
        )

    if kinds:
        return (
            f"What is the role and behavior of `{kinds}` "
            f"in {service} ({topic})?"
        )

    return f"Explain how {topic} works in {service}."


def build_dataset(
    seed: int = 42,
    max_per_service: int = 60,
) -> None:

    rng = random.Random(seed)

    by_service: dict[str, list] = {
        "kubernetes": [],
        "aws-eks": [],
        "etcd-cncf": [],
    }

    # Collect eligible chunks from the raw documentation.
    for md_file in sorted(RAW_DIR.rglob("*.md")):

        for ch in chunk_markdown_file(md_file, RAW_DIR):

            if (
                ch.metadata.service in by_service
                and len(ch.content) >= 350
                and "whatsnext"
                not in ch.metadata.section_hierarchy.lower()
            ):

                by_service[ch.metadata.service].append(ch)

    examples: list[dict] = []

    # Generate positive and negative grading examples.
    for svc, pool in by_service.items():

        rng.shuffle(pool)

        selected = pool[:max_per_service]

        for idx, pos_ch in enumerate(selected):

            q = _question_from_chunk(
                svc,
                pos_ch.metadata.section_hierarchy,
                pos_ch.content,
            )

            distractor = pool[(idx + 7) % len(pool)]

            # Positive example.
            pos_ctx = (
                f"{pos_ch.content[:1100]}\n\n"
                f"---\n\n"
                f"{distractor.content[:900]}"
            )

            pos_out = json.dumps(
                {
                    "is_sufficient": True,
                    "confidence": 0.94,
                    "reasoning": (
                        f"Context directly documents "
                        f"{pos_ch.metadata.section_hierarchy} "
                        f"for {svc}."
                    ),
                }
            )

            examples.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": (
                                f"<question>{q}</question>\n\n"
                                f"<context>\n{pos_ctx}\n</context>"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": pos_out,
                        },
                    ]
                }
            )

            # Negative example.
            neg_a = pool[(idx + 13) % len(pool)]
            neg_b = pool[(idx + 29) % len(pool)]

            neg_ctx = (
                f"{neg_a.content[:1000]}\n\n"
                f"---\n\n"
                f"{neg_b.content[:1000]}"
            )

            neg_out = json.dumps(
                {
                    "is_sufficient": False,
                    "confidence": 0.18,
                    "reasoning": (
                        f"Retrieved chunks cover "
                        f"{neg_a.metadata.section_hierarchy}, "
                        f"not the asked {svc} feature."
                    ),
                }
            )

            examples.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": (
                                f"<question>{q}</question>\n\n"
                                f"<context>\n{neg_ctx}\n</context>"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": neg_out,
                        },
                    ]
                }
            )

    # Shuffle and split the generated dataset.
    rng.shuffle(examples)

    split = int(len(examples) * 0.85)

    train_examples = examples[:split]
    val_examples = examples[split:]

    # Create output directory.
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write training dataset.
    (OUT_DIR / "train.jsonl").write_text(
        "\n".join(
            json.dumps(x)
            for x in train_examples
        ) + "\n",
        encoding="utf-8",
    )

    # Write validation dataset.
    (OUT_DIR / "val.jsonl").write_text(
        "\n".join(
            json.dumps(x)
            for x in val_examples
        ) + "\n",
        encoding="utf-8",
    )

    print(
        f"Generated {len(train_examples)} train "
        f"and {len(val_examples)} val grading examples "
        f"in {OUT_DIR}/"
    )


if __name__ == "__main__":
    build_dataset()
