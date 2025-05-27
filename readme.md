# Meta-Optimized Omniformer

**Hyperparameter Tuning for the OMICRON Pipeline using Meta Learning (LIGO O3a Run)**

## Overview

This project introduces a novel meta-learning approach to hyperparameter tuning for the [OMICRON pipeline](https://gw-openscience.org/omicron/), a tool used in gravitational wave (GW) data analysis. The primary goal is to automate and optimize the selection of hyperparameters used in detecting non-Gaussian noise transients in LIGO O3a strain data.

## Motivation

The OMICRON pipeline requires careful manual tuning of up to 10–15 hyperparameters, such as frequency thresholds, SNR cutoffs, Q-range, and PSD length. This process can be cumbersome and subjective. Our project leverages deep learning and meta-learning techniques to automate this tuning process for improved noise transient detection accuracy.

## Key Contributions

- 📊 Constructed a dataset from OMICRON runs using varying hyperparameter configurations.
- 🧠 Employed classification outputs (from Random Forest / KarooGP models) to train a transformer-based model.
- 🔁 Introduced a meta-optimizer that tunes the transformer's weights via a hypernetwork.
- ⚙️ Created a 3-way optimization pipeline involving:
  1. Transformer-based modeling
  2. Hypernetwork-based parameter control
  3. Meta-optimizer feedback tuning

## Methodology

1. **Data Collection:** OMICRON is run over multiple hyperparameter sets to generate output files (`.hdf5`, `.csv`, `.root`).
2. **Classification:** Noise transients are classified using ML models like Random Forest and KarooGP.
3. **Transformer Training:** A transformer-based model (Omniformer) is trained on classification reports.
4. **Hypernetwork Optimization:** A hypernetwork dynamically tunes model weights during training.
5. **Meta-Learning Control:** A meta-optimizer evaluates performance and refines tuning.

## Why Meta-Learning?

Meta-learning ("learning to learn") allows the system to improve its tuning ability across multiple configurations and datasets, especially effective for:
- Handling non-Gaussian noise
- Reducing manual trial-and-error
- Enhancing generalization across LIGO runs

## 🔍 References

- [Deep Filtering for Gravitational Waves (arXiv:2408.05151)](https://arxiv.org/pdf/2408.05151)
- [Meta-learning in Neurocomputing (DOI:10.1016/j.neucom.2014.12.100)](https://doi.org/10.1016/j.neucom.2014.12.100)
- [Meta-Learning Survey (IEEE)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9961879)
- [Meta-Learning Techniques (Springer)](https://link.springer.com/content/pdf/10.1007/s10994-018-5710-8.pdf)

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **LIGO Open Science Center (GWOSC):** For access to gravitational wave data.
- **OMICRON Pipeline Developers:** For the core software used in noise transient detection.
- **Gravitational Wave Research Community:** For their foundational work in transient classification and deep filtering.
- **Researchers in Meta-Learning and Transformers:** Whose models inspired this pipeline optimization framework.
- **Academic Mentors & Collaborators:** For guidance, technical input, and project inspiration.

---
