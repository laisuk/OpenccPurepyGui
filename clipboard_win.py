import ctypes
import contextlib
from ctypes import wintypes, sizeof, c_wchar_p, c_size_t

# Constants
GMEM_MOVEABLE = 0x0002
CF_UNICODETEXT = 13

# Load DLLs
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Define ctypes functions and constants
OpenClipboard = user32.OpenClipboard
OpenClipboard.argtypes = [wintypes.HWND]
OpenClipboard.restype = wintypes.BOOL

CloseClipboard = user32.CloseClipboard
CloseClipboard.argtypes = []
CloseClipboard.restype = wintypes.BOOL

EmptyClipboard = user32.EmptyClipboard
EmptyClipboard.argtypes = []
EmptyClipboard.restype = wintypes.BOOL

SetClipboardData = user32.SetClipboardData
SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
SetClipboardData.restype = wintypes.HANDLE

GetClipboardData = user32.GetClipboardData
GetClipboardData.argtypes = [wintypes.UINT]
GetClipboardData.restype = wintypes.HANDLE

GlobalAlloc = kernel32.GlobalAlloc
GlobalAlloc.argtypes = [wintypes.UINT, c_size_t]
GlobalAlloc.restype = wintypes.HGLOBAL

GlobalLock = kernel32.GlobalLock
GlobalLock.argtypes = [wintypes.HGLOBAL]
GlobalLock.restype = wintypes.LPVOID

GlobalUnlock = kernel32.GlobalUnlock
GlobalUnlock.argtypes = [wintypes.HGLOBAL]
GlobalUnlock.restype = wintypes.BOOL


# Helper function to print the last error
def print_last_error():
    error_code = ctypes.GetLastError()
    error_message = ctypes.FormatError(error_code)
    print(f"Error code: {error_code}, message: {error_message}")


@contextlib.contextmanager
def clipboard():
    """Context manager for handling clipboard operations."""
    success = OpenClipboard(None)
    if not success:
        print("Failed to open clipboard")
        print_last_error()
        raise RuntimeError("Unable to open clipboard")

    try:
        yield
    finally:
        CloseClipboard()


def set_clipboard_text(text):
    """Sets the given text to the clipboard."""
    with clipboard():
        if not EmptyClipboard():
            print("Failed to empty clipboard")
            print_last_error()
            return False

        text = str(text)
        buffer_size = (len(text) + 1) * sizeof(ctypes.c_wchar)
        handle = GlobalAlloc(GMEM_MOVEABLE, buffer_size)
        if not handle:
            print("Failed to allocate global memory")
            print_last_error()
            return False

        data = GlobalLock(handle)
        if not data:
            print("Failed to lock global memory")
            print_last_error()
            kernel32.GlobalFree(handle)  # Free the allocated memory
            return False

        try:
            ctypes.memmove(data, c_wchar_p(text), buffer_size)
        finally:
            unlock_result = GlobalUnlock(handle)
            if unlock_result == 0 and ctypes.GetLastError() != 0:
                print("Failed to unlock global memory")
                print_last_error()

        if not SetClipboardData(CF_UNICODETEXT, handle):
            print("Failed to set clipboard data")
            print_last_error()
            kernel32.GlobalFree(handle)  # Free the allocated memory
            return False

        return True


def get_clipboard_text():
    """Gets text from the clipboard."""
    with clipboard():
        handle = GetClipboardData(CF_UNICODETEXT)
        if not handle:
            print("Failed to get clipboard data")
            print_last_error()
            return ""

        data = GlobalLock(handle)
        if not data:
            print("Failed to lock global memory")
            print_last_error()
            return ""

        try:
            text = c_wchar_p(data).value
            return text if text else ""
        finally:
            unlock_result = GlobalUnlock(handle)
            if unlock_result == 0 and ctypes.GetLastError() != 0:
                print("Failed to unlock global memory")
                print_last_error()


# Example usage of the functions
if __name__ == "__main__":
    # Set clipboard text
    if set_clipboard_text("Hello, Clipboard!"):
        print("Clipboard text set successfully.")
    else:
        print("Failed to set clipboard text.")

    # Get clipboard text
    clipboard_text = get_clipboard_text()
    print(f"Clipboard Text: {clipboard_text}")
