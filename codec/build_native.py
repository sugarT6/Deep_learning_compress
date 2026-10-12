"""Build the single-threaded C++ range kernel: python -m codec.build_native."""

import argparse
import json
import os
import shlex
import subprocess
import tempfile

from ._native_range import load_native_kernel, native_library_path, native_source_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild an existing library")
    args = parser.parse_args()
    target = native_library_path()
    try:
        if not target.exists() or args.force:
            target.parent.mkdir(parents=True, exist_ok=True)
            compiler = shlex.split(os.environ.get("CXX", "c++"))
            with tempfile.TemporaryDirectory(prefix="build-", dir=target.parent) as temporary:
                output = os.path.join(temporary, target.name)
                command = compiler + ["-std=c++14", "-O3", "-DNDEBUG", "-fPIC", "-shared",
                                      str(native_source_path()), "-o", output]
                print(shlex.join(command), flush=True)
                subprocess.run(command, check=True)
                os.replace(output, target)
            target.with_suffix(".so.json").write_text(json.dumps({
                "command": command, "library": str(target),
                "compiler_version": subprocess.check_output(compiler + ["--version"], text=True),
            }, indent=2) + "\n")
        load_native_kernel()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"native build failed: {exc}\n")
    print(f"Native range coder ready (single-threaded): {target}")


if __name__ == "__main__":
    main()
