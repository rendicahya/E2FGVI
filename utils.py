from pathlib import Path


def count_files(path: Path, recursive=True, extension=None):
    pattern = "**/*" if recursive else "*"

    if extension is not None:
        pattern += extension

    return sum(1 for f in path.glob(pattern) if f.is_file())
