from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="travelgym",
    version="1.0.0",
    author="Salesforce AI Research",
    author_email="cqian@salesforce.com",
    description="A Gymnasium environment for travel planning preference elicitation simulation using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SalesforceAIResearch/UserBench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "together",
        "google-genai",
        "boto3",
        "openai",
        "pyarrow",
        "fastparquet",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    include_package_data=True,
    package_data={
        "travelgym": ["data/*.json"],
    },
) 