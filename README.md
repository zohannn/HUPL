# HUPL: a Human-like Upper-limb Posture Learner
This is a set of scripts based on the functions in the file `hupl.py` for the incremental learning of human-like target postures for the humanoid robot [***ARoS***](https://github.com/zohannn/aros_description). A human learning-by-doing cycle is mimicked by the introduction of a Variable-kernel Similarity Metric (VSM) that provides a measure of distance between different situations of a workspace. The scripts `training_vsm.py` and `predicting_vsm.py` are used by the [Motion Manager](https://github.com/zohannn/motion_manager) to train on collected optimal data and predict human-like arm configurations on novel situations, respectively. The description of the leaner and some important results have been published in G. Gulletta, W. Erlhagen and E. Bicho, "Continual Learning of Human-like Arm Postures," 2021 IEEE International Conference on Development and Learning (ICDL), 2021, pp. 1-6, [doi: 10.1109/ICDL49984.2021.9515565](https:://doi.org/10.1109/ICDL49984.2021.9515565).  
## Overview
The purpose of this document is to provide a brief introduction of the scripts, while more technical details are described in the [Wiki pages](https://github.com/zohannn/HUPL/wiki).

## Download

```Bash
cd /home/${USER}
git clone https://github.com/zohannn/HUPL.git
```
