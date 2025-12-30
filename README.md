# Modeling and Control of Concentric Tube Continuum Robots

## Overview
This repository builds on an upstream CTCR modeling implementation and adds control tooling, including wrappers for the CTCR model with **CasADi** for **MPC** formulation and testing. 
It is **based on** the original source code from:
https://github.com/TIMClab-CAMI/Modeling-and-Control-of-Concentric-Tube-Continuum-Robots

## Citation
If you use this work (or the upstream code) in academic or technical communications, please cite the original work and source code associated with:

> Quentin Boyer, Sandrine Voros, Pierre Roux, François Marionnet, Kanty Rabenorosoa and M. Taha Chikhaoui,  
> "On High Performance Control of Concentric Tube Continuum Robots Through Parsimonious Calibration,"  
> IEEE Robotics and Automation Letters, doi: 10.1109/LRA.2024.3455906

### Acknowledgement to the upstream authors
Many thanks to the original authors for their high-performance exact CTCR model implementation.

## Our Contributions
This work extends the upstream CTCR modeling code with Model Predictive Control (MPC) tooling and a few practical improvements.

### 1) CasADi integration for MPC prototyping
- Integrated **CasADi** to formulate and test MPC controllers using the CTCR exact model.

### 2) MPC with the exact CTCR model
- Implemented and tested **MPC using the exact model** provided in the upstream repository, leveraging its optimized and fast implementation.

### 3) Jacobian-based MPC
- Implemented and tested **Jacobian-based MPC**, using Jacobian information to improve optimization performance and control behavior.

### 4) MPC with exact model as an IVP with augmented decision variables
- Implemented and tested an MPC formulation where the exact CTCR model is handled as an **Initial Value Problem (IVP)**.
- Included the **initial unknown variables** directly inside the MPC optimization problem as **decision variables**, enabling joint estimation/optimization when required.

### 5) Minor performance and convergence improvements
- Added small (modest) updates to some parts of the codebase to improve:
  - **convergence** robustness
  - **speed** of model computation


## Prerequisites
(Linux) Install the following packages: `cmake build-essential libboost-all-dev gnuplot`
```sh
sudo apt install cmake build-essential libboost-all-dev gnuplot
```

## Configure and build
Create a build folder in the repository, and run CMake in it.
```sh
mkdir build
cd build
cmake ..
```
Compile the library and the demonstation codes
```sh
make -j
```
Execute the demonstation code
```sh
demo/001_model_computation
```

---

## 📦 Included Dependencies: CasADi & IPOPT

This project bundles **prebuilt CasADi** (shared library + headers) and **prebuilt IPOPT** to simplify setup and distribution.  
You do **not** need system‑wide installations of CasADi or IPOPT — the project builds out‑of‑the‑box using the versions included in the repository.

### 🏗️ Build Environment for Prebuilt Libraries

The bundled binaries were built on the following system:

```
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.3 LTS
Release:        24.04
```

**Included versions:**

| Library | Version |
|--------|---------|
| CasADi  | 3.7.2   |
| IPOPT   | 3.11.9  |

---

## 🔧 Using Your Own CasADi/IPOPT Installation

If you already have **CasADi** and **IPOPT** installed on your system and prefer to use those instead of the bundled binaries, you can switch easily.

Just **uncomment the relevant sections** in the following CMake files:

- `CMakeLists.txt` (project root)  
- `demo/CMakeLists.txt`  
- `ctcr_casadi_warpers/CMakeLists.txt`

These sections configure the project to link against your system‑installed versions rather than the prebuilt ones.

---

## Upstream source
Derived from: https://github.com/TIMClab-CAMI/Modeling-and-Control-of-Concentric-Tube-Continuum-Robots  
Many thanks to the original authors for their high-performance exact CTCR model implementation.

## Licence
This project is licensed under the **GNU GPL v3.0** (same as upstream). See the local [LICENSE](LICENSE) file.

## Contributing
Feel free to submit pull requests and use the issue tracker to start a discussion about any bugs you encounter. Please provide detailed description of the versions of your operating system, tools, and libraries for any software related bugs.
