"""Create the Geometry TempoKAN source, evidence, and figure archive."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from Datasets.prepare import sha256
from .verify import archived_checkpoints, verify_source, verify_zip


FOLDERS = ("Proposed_Method", "Datasets", "Results", "tests", "Baselines", "Experiments")
ROOT_FILES = ("README.md", "requirements.txt", "run_all.sh")


def release_files(project):
    checkpoints = archived_checkpoints(project)
    for item in ROOT_FILES:
        yield project / item
    for folder in FOLDERS:
        for path in sorted((project / folder).rglob("*")):
            relative = path.relative_to(project)
            if not path.is_file() or any(x.startswith(".") or x == "__pycache__" for x in relative.parts):
                continue
            if path.suffix in (".pyc", ".part", ".lock") or relative.parts[:2] in (
                    ("Datasets", "Cache"), ("Results", "Reproduction")):
                continue
            if relative.parts[:2] == ("Results", "Runs") and path.name == "checkpoint.pt":
                if "/".join(relative.parts[2:-1]) not in checkpoints:
                    continue
            yield path


def pack(project, output):
    verify_source(project)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".part.zip")
    checksums = []
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=7) as z:
        for path in release_files(project):
            name = path.relative_to(project).as_posix()
            z.write(path, f"Geometry_TempoKAN/{name}")
            checksums.append(f"{sha256(path)}  {name}")
        z.writestr("Geometry_TempoKAN/SHA256SUMS.txt", "\n".join(checksums) + "\n")
    verify_zip(tmp)
    tmp.replace(output)
    print(f"{output.name}  {output.stat().st_size:,} bytes  {len(checksums) + 1} files")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=Path("../Geometry_TempoKAN_Extended_Tests.zip"))
    args = p.parse_args()
    pack(args.project.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
