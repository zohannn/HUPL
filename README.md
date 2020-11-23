# HUPL: a Human-like Upper-limb Posture Learner
This is a set of scripts based on the functions in the file `hupl.py` for the incremental learning of human-like target postures for the humanoid robot [***ARoS***](https://github.com/zohannn/aros_description). A human learning-by-doing cycle is mimicked by the introduction of a Variable-kernel Similarity Metric (VSM) that provides a measure of distance between different situations of a workspace. The scripts `training_vsm.py` and `predicting_vsm.py` are used by the [Motion Manager](https://github.com/zohannn/motion_manager) to train on collected optimal data and predict human-like arm configurations on novel situations, respectively.  
## Overview
The purpose of this document is to provide a brief introduction of the scripts, while more technical details are described in the [Wiki pages](https://github.com/zohannn/HUPL/wiki).

## Download

```Bash
cd /home/${USER}
git clone https://github.com/zohannn/HUPL.git
```
