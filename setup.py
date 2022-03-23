"""Setup."""
import setuptools

with open("readme.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

setuptools.setup(
    name="ap",
    version="0.0.1",
    author="ling",
    author_email="lingjie@u.nus.edu",
    description="asset pricing models",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/lingjie00/asset_pricing",
    project_urls={
        "Bug Tracker": "https://github.com/lingjie00/asset_pricing/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "ap"},
    packages=setuptools.find_packages(where="ap"),
    python_requires=">=3.8"
)
