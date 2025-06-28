Based on single-cell-derived functional features, we constructed multiple machine learning models
for subtype classification using bulk transcriptomic data. Specifically, functional features 
distinguishing AM and CM were first identified from single-cell data, and four representative gene
sets were established accordingly. These gene sets were then used to extract input features from 
bulk data to train subtype classification models using various machine learning algorithms.

## Getting Started

Ensure you have the following prerequisites installed:
- Python >= 3.9.x
- NumPy == 1.26.4
- Pandas == 2.2.2
- Scikit-learn == 1.5.2

You can install the required packages using `pip`:

```bash
pip install numpy scikit-learn
```

or use the following command to install all dependencies：

```bash
pip install -r requirements.txt
```

### Installation

clone the repository:

```bash
git clone https://github.com/bowei-color/Melanoma.git
```

Create a virtual environment using conda:

```bash
conda create --name environment_for_aetrans python=3.11.9
```

Activate the virtual environment

```bash
conda activate myenvironment
```

