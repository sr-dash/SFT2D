from setuptools import setup, find_packages
#import versioneer

setup(
    name="sft2d",
    description="A modular Python package for simulating solar surface flux transport.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Soumyaranjan Dash",
    author_email="dash.soumya922@gmail.com",
    url="https://github.com/sr-dash/sft2d",
    license="CC0-1.0",
    packages=["sft2d", "sft2d.src", "sft2d.analysis"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "astropy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CC0-1.0",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
)
