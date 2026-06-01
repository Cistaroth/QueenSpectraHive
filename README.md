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
Queen Bee or Not Queen Bee, that is the question.
</p>

## Dataset

This project uses the following dataset found on Kaggle: [Dataset](https://www.kaggle.com/datasets/annajyang/beehive-sounds)


## How to run
> [!IMPORTANT]
> To run the program, we provide two methods: Docker vs Manual. Refer to relevant sections.

> [!NOTE]
> The Pretrained Fusion LSTM is available, if you wish to make use of a different model, modify the code, especially src/api/routers/spectral_inference/inference_pipeline and insert your model into the trained_models/ directory.

### → Docker

We assume installation of Docker Desktop or an equivalent. For installation guides refer to:
- [Docker Installation Guide](https://www.docker.com/get-started/).

From here, run the following commands:
```
# Clone repository
git clone https://github.com/Cistaroth/QueenSpectraHive.git

# Move into folder
cd "QueenSpectraHive"

# Set up environment files - Insert values into .env afterwards
cp .env.example .env

# Run docker
docker compose up --build

# The service now runs on HOST:PORT
```


### → Manually

Linux is the preferred system to run on, MacOS and Windows are supported but not recommended. Use Windows Subsystem Linux (WSL) if needed.

In order to run, python and Astral UV package manager must be installed. For instructions refer to: 
 - [[Python Installation Instructions]](https://www.python.org/downloads/) 
- [[Astral UV Installation Instructions]](https://docs.astral.sh/uv/getting-started/installation/)

From here, run the following commands:
```
# Clone repository
git clone https://github.com/Cistaroth/QueenSpectraHive.git

# Move into folder
cd "QueenSpectraHive"

# Install all required libraries, first option is for cpu, second option is for gpu
uv sync --extra gpu
uv sync --extra cpu

# Set up environment files - Insert values into .env afterwards
cp .env.example .env

# Activate virtual environment (Linux/MacOS)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate 

# For activating the inference API endpoint, run main.py
uv run "src/main.py"

# The service now runs on HOST:PORT

# For running any other scripts
uv run "SCRIPT-PATH-HERE"
```

## Credits
[@V1b1ngC0w](https://github.com/V1b1ngC0w) [@Emilvandermeer](https://github.com/Emilvandermeer) [@SyntaxSculptor1](https://github.com/SyntaxSculptor1) [@Cistaroth](https://github.com/Cistaroth)