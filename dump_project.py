from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(r"C:\Users\A\Desktop\cl\paper-contact")

OUTPUT_FILE = Path(
    r"C:\Users\A\Desktop\cl\paper-contact_FULL.txt"
)


# ============================================================
# DIRECTORIES TO IGNORE
# ============================================================

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "build",
    "dist",
}


# ============================================================
# BINARY FILE EXTENSIONS TO IGNORE
# ============================================================

EXCLUDED_EXTENSIONS = {
    # Python compiled
    ".pyc",
    ".pyo",

    # Executables / libraries
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",

    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".ico",
    ".tiff",

    # Video
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",

    # Audio
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",

    # Archives
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".bz2",

    # ML models
    ".pt",
    ".pth",
    ".onnx",
    ".safetensors",
    ".ckpt",

    # Databases
    ".db",
    ".sqlite",
    ".sqlite3",
}


# ============================================================
# CHECK TEXT FILE
# ============================================================

def is_text_file(path: Path) -> bool:

    try:

        with open(path, "rb") as f:
            data = f.read(8192)

        # Empty files are considered text
        if not data:
            return True

        # NULL byte usually means binary
        if b"\x00" in data:
            return False

        encodings = [
            "utf-8",
            "utf-16",
            "cp1256",
            "cp1252",
            "latin-1",
        ]

        for encoding in encodings:

            try:
                data.decode(encoding)
                return True

            except UnicodeDecodeError:
                pass

        return False

    except Exception:
        return False


# ============================================================
# READ FILE
# ============================================================

def read_file(path: Path) -> str:

    encodings = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp1256",
        "cp1252",
        "latin-1",
    ]

    for encoding in encodings:

        try:

            return path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError:

            continue

        except Exception as e:

            return f"[ERROR READING FILE: {e}]"

    try:

        return path.read_text(
            encoding="utf-8",
            errors="replace"
        )

    except Exception as e:

        return f"[ERROR READING FILE: {e}]"


# ============================================================
# CHECK IF PATH SHOULD BE IGNORED
# ============================================================

def should_ignore(path: Path) -> bool:

    for part in path.parts:

        if part in EXCLUDED_DIRS:

            return True

    return False


# ============================================================
# COLLECT ALL TEXT / CODE FILES
# ============================================================

def collect_files():

    files = []

    for path in ROOT_DIR.rglob("*"):

        if not path.is_file():
            continue

        if should_ignore(path):
            continue

        if path.suffix.lower() in EXCLUDED_EXTENSIONS:
            continue

        if not is_text_file(path):
            continue

        files.append(path)

    return sorted(files)


# ============================================================
# BUILD DIRECTORY TREE
# ============================================================

def build_tree():

    """
    Creates a visual directory/file tree.

    Example:

    paper-contact/
    ├── app/
    │   ├── main.py
    │   └── db/
    │       └── repository.py
    └── README.md
    """

    lines = []

    root_name = ROOT_DIR.name

    lines.append(root_name + "/")

    def walk(directory: Path, prefix=""):

        try:

            entries = [
                p
                for p in directory.iterdir()
                if not should_ignore(p)
            ]

        except PermissionError:

            lines.append(
                prefix + "└── [PERMISSION DENIED]"
            )

            return

        # Remove excluded binary files
        filtered_entries = []

        for p in entries:

            if p.is_file():

                if p.suffix.lower() in EXCLUDED_EXTENSIONS:
                    continue

                if not is_text_file(p):
                    continue

            filtered_entries.append(p)

        entries = sorted(
            filtered_entries,
            key=lambda x: (
                x.is_file(),
                x.name.lower()
            )
        )

        for index, entry in enumerate(entries):

            is_last = index == len(entries) - 1

            connector = "└── " if is_last else "├── "

            lines.append(
                prefix +
                connector +
                entry.name +
                ("/" if entry.is_dir() else "")
            )

            if entry.is_dir():

                new_prefix = (
                    prefix + "    "
                    if is_last
                    else prefix + "│   "
                )

                walk(
                    entry,
                    new_prefix
                )

    walk(ROOT_DIR)

    return "\n".join(lines)


# ============================================================
# GENERATE OUTPUT
# ============================================================

def generate_output():

    print()
    print("=" * 70)
    print("PAPER-CONTACT PROJECT DUMPER")
    print("=" * 70)
    print()

    if not ROOT_DIR.exists():

        print("ERROR: Project directory does not exist.")
        print()
        print(ROOT_DIR)

        return

    print("Project:")
    print(ROOT_DIR)
    print()

    # --------------------------------------------------------
    # Collect files
    # --------------------------------------------------------

    files = collect_files()

    print(
        f"Found {len(files)} text/code files."
    )

    print()

    # --------------------------------------------------------
    # Build tree
    # --------------------------------------------------------

    print("Building directory tree...")

    tree = build_tree()

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    print("Writing output...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="\n"
    ) as out:

        # ====================================================
        # HEADER
        # ====================================================

        out.write(
            "PAPER-CONTACT - COMPLETE PROJECT DUMP\n"
        )

        out.write(
            "=" * 100
        )

        out.write("\n\n")

        out.write(
            f"ROOT PATH:\n{ROOT_DIR}\n\n"
        )

        out.write(
            f"TOTAL TEXT/CODE FILES: {len(files)}\n\n"
        )

        # ====================================================
        # DIRECTORY TREE
        # ====================================================

        out.write(
            "=" * 100
        )

        out.write("\n")

        out.write(
            "PROJECT DIRECTORY TREE"
        )

        out.write("\n")

        out.write(
            "=" * 100
        )

        out.write("\n\n")

        out.write(tree)

        out.write("\n\n")

        # ====================================================
        # FILE CONTENTS
        # ====================================================

        out.write(
            "=" * 100
        )

        out.write("\n")

        out.write(
            "FILE CONTENTS"
        )

        out.write("\n")

        out.write(
            "=" * 100
        )

        out.write("\n\n")

        # ----------------------------------------------------
        # Every file
        # ----------------------------------------------------

        for index, file_path in enumerate(files, 1):

            relative_path = file_path.relative_to(
                ROOT_DIR
            )

            print(
                f"[{index}/{len(files)}] "
                f"{relative_path}"
            )

            out.write("\n")

            out.write(
                "=" * 100
            )

            out.write("\n")

            out.write(
                f"FILE #{index}\n"
            )

            out.write(
                f"RELATIVE PATH: {relative_path}\n"
            )

            out.write(
                f"FULL PATH: {file_path}\n"
            )

            out.write(
                "=" * 100
            )

            out.write("\n\n")

            try:

                content = read_file(
                    file_path
                )

                out.write(content)

                if not content.endswith("\n"):
                    out.write("\n")

            except Exception as e:

                out.write(
                    f"[ERROR READING FILE: {e}]\n"
                )

            out.write("\n")

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    print()
    print("Output file:")
    print(OUTPUT_FILE)
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    generate_output()