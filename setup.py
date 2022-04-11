import io
import os

from setuptools import find_packages, setup


__project__ = ""
__version__ = ""
exec(open(os.path.join("pyqtorch", "_version.py")).read())

DESCRIPTION = ""
URL = ""
EMAIL = ""
AUTHOR = ""
REQUIRES_PYTHON = ">=3.8.0"

REQUIRED = ["torch", "numpy"]

EXTRAS = { }

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "..", "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=__project__,
    version=__version__,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["*.tests.*", "tests.*", "*.tests"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        "License :: Other/Proprietary License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    zip_safe=True,
)
