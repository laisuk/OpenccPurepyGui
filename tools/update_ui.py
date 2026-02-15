import os
import subprocess
from pathlib import Path
from datetime import datetime

from typing import List

RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
RESET = "\033[0m"


def get_last_write_time(file_path: Path):
    return datetime.fromtimestamp(file_path.stat().st_mtime)


def format_datetime(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def update_if_newer(source: Path, target: Path, command: List[str]) -> None:
    if not source.exists():
        print(f"{source.name} not found.")
        return

    print(f"{source.name:<15}: {format_datetime(get_last_write_time(source))}")

    if not target.exists():
        print(f"{target.name:<15}: (missing)")
        print(f"{YELLOW}Target missing, generating...{RESET}")
        subprocess.run(command)
        print(f"{BLUE}{target.name} generated.{RESET}")
        return

    print(f"{target.name:<15}: {format_datetime(get_last_write_time(target))}")

    is_newer = get_last_write_time(source) > get_last_write_time(target)

    print(f"{source.name} newer than {target.name}: ", end="")
    print(f"{GREEN}{is_newer}{RESET}" if is_newer else f"{RED}{is_newer}{RESET}")

    if is_newer:
        subprocess.run(command)
        print(f"{BLUE}{target.name} updated.{RESET}")
    else:
        print(f"{BLUE}No update needed.{RESET}")


def main():
    # tools/ directory
    tools_dir = Path(__file__).parent
    # project root directory (one level up)
    project_root = tools_dir.parent

    os.chdir(project_root)

    update_if_newer(
        project_root / "form.ui",
        project_root / "ui_form.py",
        ["pyside6-uic", "form.ui", "-o", "ui_form.py"]
    )

    update_if_newer(
        project_root / "resource.qrc",
        project_root / "resource_rc.py",
        ["pyside6-rcc", "resource.qrc", "-o", "resource_rc.py"]
    )


if __name__ == "__main__":
    main()
