""" post_install.py a post install script to be called on the target machine after the installation """

import hashlib
import logging
import json
import os
import platform
import subprocess
import zipfile
import ctypes

import requests

from src.consts import APP_DATA
from src.camera import CAMERA_NAME

LIBRARY_NAMES = ["UnityCaptureFilter32.dll", "UnityCaptureFilter64.dll"]
CACHE_DIR = os.path.join(APP_DATA, ".cache")
PACKAGE_DIR = os.path.join(CACHE_DIR, "UnityCapture")
INTEGRITY_FILE = os.path.join(CACHE_DIR, ".integrity")


def ensure_admin():
    if platform.system() != "Windows":
        raise OSError("This script is only supported on Windows")
    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise PermissionError("Please run this script as an administrator")


def is_first_install():
    tracker = os.path.join(APP_DATA, ".postinstall")
    return not bool(os.path.exists(tracker))


def save_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    with open(INTEGRITY_FILE, "r") as f:
        lib_hashes = json.load(f)

    lib_hashes[file_path] = file_hash

    with open(INTEGRITY_FILE, "w") as f:
        json.dump(lib_hashes, f)


def validate_hash(file_path):
    base_name = os.path.basename(file_path)
    logging.info(f"\tValidating hash for {base_name}")
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    with open(INTEGRITY_FILE, "r") as f:
        lib_hashes = json.load(f)

    try:
        valid = lib_hashes[file_path] == file_hash
        if not valid:
            logging.warning(f"\t\tHash for {base_name} is invalid")
        else:
            logging.info(f"\t\tHash for {base_name} is valid")
        return valid
    except KeyError:
        logging.warning(f"\t\tHash for {base_name} not found")
        return False


def get_packages():
    PACKAGE_URL = "https://github.com/schellingb/UnityCapture/archive/master.zip"

    # download the package
    logging.info("\tDownloading UnityCapture package...")
    response = requests.get(PACKAGE_URL)
    response.raise_for_status()
    logging.info("\t\tDownloaded UnityCapture package.")

    # save the package to .cache in appdata
    logging.info("\tExtracting UnityCapture package...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    PACKAGE_ZIP = os.path.join(CACHE_DIR, "UnityCapture.zip")

    with open(PACKAGE_ZIP, "wb") as f:
        f.write(response.content)

    # extract the package
    with zipfile.ZipFile(PACKAGE_ZIP, "r") as z:
        z.extractall(PACKAGE_DIR)
    logging.info("\t\tExtracted UnityCapture package.")

    # write a hash of the library files:
    for lib in LIBRARY_NAMES:
        lib_path = os.path.join(PACKAGE_DIR, "UnityCapture-master", "Install", lib)
        save_hash(lib_path)


def install_packages():
    logging.info("Installing UnityCapture lib...")
    if not os.path.exists(INTEGRITY_FILE):
        with open(INTEGRITY_FILE, "w") as f:
            json.dump({}, f)

    integrity = all(
        validate_hash(os.path.join(PACKAGE_DIR, "UnityCapture-master", "Install", lib))
        for lib in LIBRARY_NAMES
    )

    if not integrity:
        get_packages()

    # install the package
    logging.info("\tRegistering UnityCapture libs...")
    LIB_DIR = os.path.join(PACKAGE_DIR, "UnityCapture-master", "Install")
    for lib in LIBRARY_NAMES:
        subprocess.run(
            [
                "regsvr32",
                os.path.join(LIB_DIR, lib),
                f"/i:UnityCaptureName='{CAMERA_NAME}'",
                "/s",
            ]
        )
    logging.info("\t\tRegistered UnityCapture libs.")
    logging.info("Installed UnityCapture libs.")


def main():
    ensure_admin()
    install_packages()


if __name__ == "__main__":
    main()
