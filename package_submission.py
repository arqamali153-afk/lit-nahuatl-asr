"""
package_submission.py

Zips the ./src directory into submission.zip at the project root, matching
the runtime's expected layout. IMPORTANT: arcname is relative to SRC_DIR
(not PROJECT_ROOT) -- confirmed by every successful smoke test log that
the runtime already extracts the zip into ./src/ itself, so the zip's own
entries must be flat (main.py, weights/...), NOT prefixed with src/, or
you get a double-nested ./src/src/main.py that breaks the run. Do not
"fix" this back to PROJECT_ROOT -- that's the bug this script already
had and already fixed once.

Excludes unused fine-tuned adapter weights (not in the production routing
path) and Python cache files from the archive.

Usage:
    python package_submission.py
"""

import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
OUTPUT_ZIP = os.path.join(PROJECT_ROOT, "submission.zip")

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "nhi_adapter_finetuned",  # not in the production routing path -- keep local only
}
EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".pyd"}


def main():
    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f"Expected a src/ directory at {SRC_DIR}")

    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)

    file_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(SRC_DIR):
            # Prune excluded directories in-place so os.walk skips them entirely.
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]

            for fname in files:
                if any(fname.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
                    continue

                full_path = os.path.join(root, fname)
                # Relative to SRC_DIR (NOT PROJECT_ROOT) -- keeps entries
                # flat, matching the runtime's own ./src extraction target.
                # zipfile.ZipFile.write() already normalizes os.sep to '/'
                # internally, but the explicit replace is a harmless
                # defensive no-op.
                arcname = os.path.relpath(full_path, SRC_DIR).replace(os.sep, "/")
                zf.write(full_path, arcname)
                file_count += 1

    print(f"[INFO] Packaged {file_count} files into {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()