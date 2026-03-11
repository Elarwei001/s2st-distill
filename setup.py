from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="s2st-distill",
    version="0.1.0",
    author="S2ST-Distill Contributors",
    description="Distill multilingual S2ST models for on-device deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Elarwei001/s2st-distill",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime>=1.16.0",
            "coremltools>=7.0",
        ],
        "eval": [
            "sacrebleu>=2.3.0",
            "resemblyzer>=0.1.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "s2st-benchmark=scripts.benchmark:main",
            "s2st-evaluate=scripts.evaluate:main",
        ],
    },
)
