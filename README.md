# ZhoConvertPyQtRs

**PySide6 Linux Notes:**

Ensure the xcb-cursor0 library is installed using:

`sudo apt-get install libxcb-cursor0`

Verify your Qt installation with:

`sudo apt-get install --reinstall qt6-default`

Check library paths:

`export LD_LIBRARY_PATH=/path/to/libxcb-cursor0:$LD_LIBRARY_PATH
`