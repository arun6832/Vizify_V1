from setuptools import setup, find_packages

setup(
    name="vizify",
    version="0.1.3",
    author="Arun M",
    author_email="arunpappulli@gmail.com",
    description="An automated data visualization package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arun6832/vizify",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "missingno",
        "wordcloud"
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
