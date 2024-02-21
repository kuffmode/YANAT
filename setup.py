from setuptools import setup, find_packages

DESCRIPTION = "Yet Another Network Analysis Toolkit (Yanat)"
with open("docs/README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(name="Yanat",
      version="0.0.5",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author='Kayson Fakhar, Shrey Dixit',
      author_email='kayson.fakhar@gmail.com, shrey.akshaj@gmail.com',
      url="https://github.com/kuffmode/YANAT",
      packages=find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3.10",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering"],
      python_requires='>=3.8',
      install_requires=["msapy>=1.5"],
      include_package_data=True)