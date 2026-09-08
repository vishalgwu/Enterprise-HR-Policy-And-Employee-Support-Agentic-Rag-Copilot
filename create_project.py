r"""Scaffold the FastAPI application layout described in the README.

Idempotent: creating an existing folder or file is a no-op, and no file that
already has content is ever truncated. data/private_kb/ and the hr/ virtualenv
already exist and are left alone.

    hr\Scripts\python.exe create_project.py
"""

from pathlib import Path

root = Path(__file__).resolve().parent

folders = [
    "app/api",
    "app/agent",
    "app/core",
    "app/rag",
    "app/services",
    "data",
    "scripts",
    "templates",
    "static",
    "uploads",
    "tests",
]

# Package markers, so `from app.core.config import ...` resolves. tests/ is
# deliberately NOT a package: pyproject puts tests/ on sys.path so the suite can
# `import conftest`, and an __init__.py there breaks that.
packages = [
    "app",
    "app/api",
    "app/agent",
    "app/core",
    "app/rag",
    "app/services",
    "scripts",
]

files = [
    "app/main.py",
    "ingest_sample_kb.py",
    "requirements.txt",
    "run.py",
    ".env",
]

# Folders git would otherwise drop, since it does not track empty directories.
keep = [
    "templates",
    "static",
    "uploads",
]


def touch(path: Path) -> bool:
    """Create an empty file. Returns True if it did not exist before."""
    if path.exists():
        return False
    path.touch()
    return True


created = []

for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)

for package in packages:
    if touch(root / package / "__init__.py"):
        created.append(f"{package}/__init__.py")

for file in files:
    if touch(root / file):
        created.append(file)

for folder in keep:
    if touch(root / folder / ".gitkeep"):
        created.append(f"{folder}/.gitkeep")

print(f"Folders ensured: {len(folders)}")
if created:
    print("Created:")
    for name in created:
        print(f"  {name}")
else:
    print("Nothing to create -- structure already present.")
