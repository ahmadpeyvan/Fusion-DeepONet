# 🚀 Fusion-DeepONet

Fusion-DeepONet is a data-efficient neural operator designed for **geometry-dependent hypersonic and supersonic flow fields**.  
This repository contains the full code base used in the paper:

> **“Fusion-DeepONet: A data-efficient neural operator for geometry-dependent hypersonic and supersonic flows”**  
> *Journal of Computational Physics, 2026*  
> [Read the paper](https://www.sciencedirect.com/science/article/pii/S0021999125007144)

All datasets required to reproduce the results are publicly available.

---

## 📁 Repository Structure

The `src/` directory includes three main problem setups:

- **Semi_ellipse**
- **Capsule**
- **Convergent_Divergent_Nozzle**

Each folder contains the complete training, inference, and post-processing scripts corresponding to the problem configurations used in the paper.

---

## 📦 Downloading the Dataset

All datasets are hosted on Zenodo:

👉 **Download Data:** https://doi.org/10.5281/zenodo.17603114

After downloading `data.zip`, you will find three folders:

Semi_ellipse/  
Capsule/  
Convergent_Divergent_Nozzle/

Each folder contains the simulation cases and training datasets required by the scripts in `src/`.

Place the data into the matching directories under `src/` following the instructions below.

---

# 🧩 How to Run Each Problem

---

## 🔷 1. Semi-Ellipse

### Required Data
Copy the following into `src/Semi_ellipse/unstructured_grid` and  `src/Semi_ellipse/structured_grid`:

- dataset_structured_grid/
- dataset_unstructured_grid/

### Training
    cd src/Semi_ellipse/unstructured_grid
    or
    cd src/Semi_ellipse/structured_grid

    python fusion_deeponet.py

### Inference
Update `inference.py` with your checkpoint, then:

    python inference.py

### Plot Basis Functions
    python inference_basis.py

---

## 🔷 2. Convergent–Divergent Nozzle

### Required Data
Copy the following:

- datasets/ → src/Convergent_Divergent_Nozzle/Training  
- cases/ → src/Convergent_Divergent_Nozzle/

### Training
    cd src/Convergent_Divergent_Nozzle/Training
    python fusion_deeponet.py

### Inference
    cd ../Inference
    python inference.py

---

## 🔷 3. Capsule

### Required Data
Copy into `src/Capsule/`:

- cases/
- mesh_folder/

Copy dataset files into:

    src/Capsule/Training/

### Training Options

Loss Type | Script
--------- | -----------------------------
No derivative enhancement | fusion_deeponet_NO_DEL.py
Least-squares derivatives | fusion_deeponet_LSD.py
Discrete derivatives | fusion_deeponet_DD.py
Autodiff derivatives | fusion_deeponet_autodiff.py

Example:

    cd src/Capsule/Training
    python fusion_deeponet_LSD.py

### Inference

For NO_DEL, LSD, DD:

    cd src/Capsule/Inference
    python inference.py

For autodiff:

    python inference_autodiff.py

### Post-processing (Heat Flux)

    cd src/Capsule/Postprocessing
    python main.py

---

# 🎥 Demo Video

Watch prediction examples:

YouTube: https://www.youtube.com/watch?v=K4TSAHu1LMw
