""" post_install.py a post install script to be called on the target machine after the installation """

import hashlib
import logging
import os
import platform
import subprocess
import zipfile

import requests

from src.consts import APP_DATA, APP_NAME


def ensure_admin():
    if platform.system() == "Windows":
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            raise PermissionError("Please run this script as an administrator")

def is_first_install():
    tracker = os.path.join(APP_DATA, ".postinstall")
    return not bool(os.path.exists(tracker))


def install_packages():
    PACKAGE_URL = "https://github.com/schellingb/UnityCapture/archive/master.zip"

    # download the package
    logging.info("Downloading UnityCapture package...")
    response = requests.get(PACKAGE_URL)
    response.raise_for_status()
    logging.info("Downloaded UnityCapture package.")

    # save the package to .cache in appdata
    logging.info("Extracting UnityCapture package...")
    cache = os.path.join(APP_DATA, ".cache")
    os.makedirs(cache, exist_ok=True)

    PACKAGE_ZIP = os.path.join(cache, "UnityCapture.zip")

    with open(PACKAGE_ZIP, "wb") as f:
        f.write(response.content)

    # extract the package
    PACKAGE_DIR = os.path.join(cache, "UnityCapture")
    with zipfile.ZipFile(PACKAGE_ZIP, "r") as z:
        z.extractall(PACKAGE_DIR)
    logging.info("Extracted UnityCapture package.")

    # install the package
    logging.info("Installing UnityCapture package...")
    LIBRARY_NAMES = ["UnityCaptureFilter32.dll", "UnityCaptureFilter64.dll"]
    LIB_DIR = os.path.join(PACKAGE_DIR, "UnityCapture-master", "Install")
    for lib in LIBRARY_NAMES:
        subprocess.run(["regsvr32", os.path.join(LIB_DIR, lib),  f"/i:UnityCaptureName='{APP_NAME} Video Capture'"])

    # create the tracker file
    tracker = os.path.join(APP_DATA, ".postinstall")
    with open(tracker, "w") as f:
        # hash the package zip file and write it to the tracker
        hasher = hashlib.sha256()
        with open(PACKAGE_ZIP, "rb") as f:
            hasher.update(f.read())
        f.write(hasher.hexdigest())
    logging.info("Installed UnityCapture package.")


def main():
    ensure_admin()
    if is_first_install():
        logging.info("First install detected...")
        install_packages()
    else:
        logging.info("Skipping post install...")

if __name__ == "__main__":
    main()
