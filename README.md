# 🐝 QueenSpectraHive 
<div align="center">

![Static Badge](https://img.shields.io/badge/github-repo-blue?logo=github)
![Static Badge](https://img.shields.io/badge/version-1.0.0-green)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

</div>

<br>
<p align= "center">
Classification of Queen Bee Presence using Tabular and Acoustic Data.
</p>

## Dataset

This project uses the following dataset found on Kaggle: [Dataset](https://www.kaggle.com/datasets/annajyang/beehive-sounds)

## How to run

Linux is the preferred system to run on, MacOS and Windows are supported but not recommended. Use Windows Subsystem Linux (WSL) if needed.

In order to run, python and Astral UV package manager must be installed. For instructions refer to: 
 - [[Python Installation Instructions]](https://www.python.org/downloads/) 
- [[Astral UV Installation Instructions]](https://docs.astral.sh/uv/getting-started/installation/)

From here, run the following commands:
```
# Clone repository
git clone https://github.com/Cistaroth/QueenSpectraHive.git

# Move into src
cd src

# Install all required libraries
uv sync

# Activate virtual environment (Linux/MacOS)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate 

# Set up environment files
cp .env.example .env

# For activating the inference API endpoint, run main.py
uv run main.py

# For running any other scripts
uv run "SCRIPT-PATH-HERE"
```

## Credits
[@V1b1ngC0w](https://github.com/V1b1ngC0w) [@Emilvandermeer](https://github.com/Emilvandermeer) [@SyntaxSculptor1](https://github.com/SyntaxSculptor1) [@Cistaroth](https://github.com/Cistaroth)