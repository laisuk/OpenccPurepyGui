#!/bin/bash

# Set working directory to the directory of the script
# shellcheck disable=SC2164
cd "$(dirname "$0")"

UiPath="./form.ui"
QrcPath="./resource.qrc"

if [ -f "$UiPath" ]; then
    lastSaveDateUI=$(stat -c %Y "$UiPath")
    echo "form.ui   : $(date -d @"$lastSaveDateUI")"

    lastSaveDateUiPy=0
    if [ -f "./ui_form.py" ]; then
        lastSaveDateUiPy=$(stat -c %Y "./ui_form.py")
        echo "ui_form.py: $(date -d @"$lastSaveDateUiPy")"
    else
        echo "ui_form.py: not found"
    fi

    echo -n "form.ui Newer than ui_form.py : "
    if [ "$lastSaveDateUI" -gt "$lastSaveDateUiPy" ]; then
        echo -e "\e[32mtrue\e[0m"
        pyside6-uic form.ui -o ui_form.py
        echo -e "\e[34mui_form.py updated.\e[0m"
    else
        echo -e "\e[31mfalse\e[0m"
        echo -e "\e[34mNo Ui update needed.\e[0m"
    fi
else
    echo -e "\e[34mform.ui not found.\e[0m"
fi

if [ -f "$QrcPath" ]; then
    lastSaveDateQrc=$(stat -c %Y "$QrcPath")
    echo "resource.qrc  : $(date -d @"$lastSaveDateQrc")"

    lastSaveDateQrcPy=0
    if [ -f "./resource_rc.py" ]; then
        lastSaveDateQrcPy=$(stat -c %Y "./resource_rc.py")
        echo "resource_rc.py: $(date -d @"$lastSaveDateQrcPy")"
    else
        echo "resource_rc.py: not found"
    fi

    echo -n "resource.qrc Newer than resource_rc.py : "
    if [ "$lastSaveDateQrc" -gt "$lastSaveDateQrcPy" ]; then
        echo -e "\e[32mtrue\e[0m"
        pyside6-rcc resource.qrc -o resource_rc.py
        echo -e "\e[34mresource_rc.py updated.\e[0m"
    else
        echo -e "\e[31mfalse\e[0m"
        echo -e "\e[34mNo Qrc update needed.\e[0m"
    fi
else
    echo -e "\e[34mresource.qrc not found.\e[0m"
fi
