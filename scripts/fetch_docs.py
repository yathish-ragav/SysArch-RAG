from pathlib import Path
import shutil
import subprocess

RAW_DIR = Path("data/raw_docs")
TMP_DIR = Path("/tmp/sysarch_doc_clones")

REPOS = [
    {
        "name": "kubernetes",
        "owner": "kubernetes",
        "repo": "website",
        "paths": ["content/en/docs/concepts", "content/en/docs/tasks"],
    },
    {
        "name": "aws-eks",
        "owner": "awsdocs",
        "repo": "amazon-eks-user-guide",
        "paths": ["doc_source"],
    },
    {
        "name": "etcd-cncf",
        "owner": "etcd-io",
        "repo": "etcd",
        "paths": ["Documentation"],
    },
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    for item in REPOS:
        name = item["name"]
        url = "https://" + "github.com/" + item["owner"] + "/" + item["repo"] + ".git"
        clone_dir = TMP_DIR / name
        dest_dir = RAW_DIR / name

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)

        print(f"Fetching [{name}] from {url} ...")
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(clone_dir)])
        run(["git", "-C", str(clone_dir), "sparse-checkout", "set", *item["paths"]])

        for rel_path in item["paths"]:
            src = clone_dir / rel_path
            if src.exists():
                for md_file in src.rglob("*.md"):
                    target = dest_dir / md_file.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(md_file, target)

        md_count = len(list(dest_dir.rglob("*.md")))
        print(f"Saved {md_count} Markdown files for [{name}]")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("\nALL DOCS FETCHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
