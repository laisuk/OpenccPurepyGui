import os
import subprocess
from pathlib import Path
from datetime import datetime

RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
RESET = "\033[0m"


def get_last_write_time(file_path):
    return datetime.fromtimestamp(file_path.stat().st_mtime)


def format_datetime(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def main():
    # Set the current working directory to the directory where the script is located
    script_root = Path(__file__).parent
    os.chdir(script_root)

    ui_path = script_root / "form.ui"
    qrc_path = script_root / "resource.qrc"

    if ui_path.exists():
        file_ui = ui_path
        last_save_date_ui = get_last_write_time(file_ui)
        print(f"form.ui   : {format_datetime(last_save_date_ui)}")

        file_ui_py = script_root / "ui_form.py"
        last_save_date_ui_py = get_last_write_time(file_ui_py)
        print(f"ui_form.py: {format_datetime(last_save_date_ui_py)}")

        print("form.ui Newer than ui_form.py : ", end="")
        is_ui_newer = last_save_date_ui > last_save_date_ui_py
        if is_ui_newer:
            print(f"{GREEN}{is_ui_newer}{RESET}")
            subprocess.run(["pyside6-uic", str(ui_path), "-o", "ui_form.py"])
            print(f"{BLUE}ui_form.py updated.{RESET}")
        else:
            print(f"{RED}{is_ui_newer}{RESET}")
            print(f"{BLUE}No Ui update needed.{RESET}")
    else:
        print("form.ui not found.")

    if qrc_path.exists():
        file_qrc = qrc_path
        last_save_date_qrc = get_last_write_time(file_qrc)
        print(f"resource.qrc  : {format_datetime(last_save_date_qrc)}")

        file_qrc_py = script_root / "resource_rc.py"
        last_save_date_qrc_py = get_last_write_time(file_qrc_py)
        print(f"resource_rc.py: {format_datetime(last_save_date_qrc_py)}")

        print("resource.qrc Newer than resource_rc.py : ", end="")
        is_qrc_newer = last_save_date_qrc > last_save_date_qrc_py
        if is_qrc_newer:
            print(f"{GREEN}{is_qrc_newer}{RESET}")
            subprocess.run(["pyside6-rcc", str(qrc_path), "-o", "resource_rc.py"])
            print(f"{BLUE}resource_rc.py updated.{RESET}")
        else:
            print(f"{RED}{is_qrc_newer}{RESET}")
            print(f"{BLUE}No Qrc update needed.{RESET}")
    else:
        print("resource.qrc not found.")


if __name__ == "__main__":
    main()
