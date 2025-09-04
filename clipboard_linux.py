import subprocess


def get_clipboard_text():
    try:
        result = subprocess.run(['xclip', '-selection', 'clipboard', '-o'],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to get clipboard text: {e.stderr}")
        return ""


def set_clipboard_text(text):
    try:
        subprocess.run(['xclip', '-selection', 'clipboard'],
                       input=text,
                       check=True,
                       text=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to set clipboard text: {e.stderr}")


if __name__ == "__main__":
    # Example usage of the functions

    # Set clipboard text
    set_clipboard_text("Hello, Clipboard!")

    # Get clipboard text
    clipboard_text = get_clipboard_text()
    print(f"Clipboard Text: {clipboard_text}")
