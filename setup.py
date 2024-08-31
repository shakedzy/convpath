import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent.resolve()

PACKAGE_NAME = "convpath"
AUTHOR = "Shaked Zychlinski"
AUTHOR_EMAIL = "shakedzy@gmail.com"
URL = "http://shakedzy.xyz/convpath"
DOWNLOAD_URL = "https://pypi.org/project/convpath/"

LICENSE = "MIT"
VERSION = (HERE / "VERSION").read_text(encoding="utf8").strip()
DESCRIPTION = "LLM conversations similarity via path comparison"
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf8")
LONG_DESC_TYPE = "text/markdown"

requirements = (HERE / "requirements.txt").read_text(encoding="utf8")
INSTALL_REQUIRES = [s.strip() for s in requirements.split("\n")]

min_minor = 9
max_minor = 12
CLASSIFIERS = [
    f"Programming Language :: Python :: 3.{str(v)}" for v in range(min_minor, max_minor+1)
]
PYTHON_REQUIRES = f">=3.{min_minor}"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    classifiers=CLASSIFIERS,
)