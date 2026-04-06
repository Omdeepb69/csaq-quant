from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="csaq-quant",
    version="0.2.3",
    author="Omdeep Borkar",
    author_email="omdeepborkar@gmail.com",
    description=(
        "Causal Salience-Aware Quantization — gradient×activation-informed "
        "interaction-graph LLM weight quantization with self-speculative decoding"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omdeepb69/csaq-quant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "tqdm>=4.66.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
        "eval": ["accelerate>=0.24.0"],
    },
    keywords=[
        "quantization", "llm", "compression", "inference",
        "causal salience", "mixed precision", "pytorch",
        "speculative decoding",
    ],
)
