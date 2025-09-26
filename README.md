# Data-Efficient Time-Dependent PDE Surrogates: Graph Neural Simulators vs. Neural Operators

[Dibyajyoti Nayak](https://scholar.google.com/citations?user=iAdGHHQAAAAJ&hl=en&oi=ao) and [Somdatta Goswami](https://scholar.google.com/citations?hl=en&user=GaKrpSkAAAAJ&view_op=list_works&sortby=pubdate)

You can find the presentation slides, outlining our approach and the results we achieved:

Slides - [GNS_vs_NOs_slides_updated](./GNS_slides_updated.pdf)

We propose Graph Neural Simulators (GNS) as a principled surrogate modeling paradigm for time-dependent PDEs. GNS leverages message-passing combined with numerical time-stepping schemes to learn PDE dynamics by modeling the instantaneous time derivatives. This design mimics traditional numerical solvers, enabling stable long-horizon rollouts and strong inductive biases that enhance generalization. 

## Results
We rigorously evaluate GNS on four canonical PDE systems: (1) 2D scalar Burgers’, (2) 2D coupled Burgers’, (3) 2D Allen–Cahn, and (4) 2D nonlinear shallow-water equations, comparing against state-of-the-art NOs including Deep Operator Network (DeepONet) and Fourier Neural Operator (FNO). Results demonstrate that GNS is markedly more data-efficient, achieving less than 1% relative $L_2$ error using only 3% of available trajectories, and exhibits dramatically reduced error accumulation over time (82.5% lower autoregressive error than FNO, 99.9% lower than DeepONet).

<!-- ## Datasets
Link to the datasets used in this work: [GNS_vs_NOs_datasets] -->

## Installation
The code for this project is written in JAX. To install the dependencies and get started, clone the repository and install the required packages:

```bash
git clone https://github.com/Centrum-IntelliPhysics/GNS_vs_NOs.git
cd GNS_vs_NOs
pip install -r requirements.txt
```

## Repository Overview
This repository contains implementations and analyses for the experiments described in the paper. There are four different PDEs in the codes folder. Except the 2D SWE example, each case has training scripts for the different frameworks employed in this study: (1) GNS, (2) FNO Autoregressive, (3) FNO Full Rollout, (4) DeepONet Autoregressive, (5) DeepONet Full Rollout, (6) TI-DeepONet, and (7) TI(L)-DeepONet. For brevity, the 2D SWE case only contains the codes for training GNS and FNO variants.

## Running Training and Inference Scripts  
All Python files for training different neural operators and GNS inference can be run using the standard `python` command. However, for training a GNS, a Torch DDP setup must be invoked from the command line using the following `torchrun` command:  

```bash
srun torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    main_multiGPU.py --total_epochs 600 --log_loss 10 --batch_size 2
```

### Citation:
Our preprint is available on [Arxiv](https://arxiv.org/abs/2509.06154). If you use this code for your research, please cite our paper.

```bash
@article{nayak2025data,
  title={Data-Efficient Time-Dependent PDE Surrogates: Graph Neural Simulators vs Neural Operators},
  author={Nayak, Dibyajyoti and Goswami, Somdatta},
  journal={arXiv preprint arXiv:2509.06154},
  year={2025}
}
```