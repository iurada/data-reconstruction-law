
<div align="center">

# A Law of Data Reconstruction for Random Features (and Beyond)

<p align="center">
  <a href="https://arxiv.org/abs/2509.22214"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square&labelColor=444444" alt="Paper arXiv"></a>
</p>

**[Leonardo Iurada](https://scholar.google.com/citations?user=S8t5OEoAAAAJ&hl)<sup>✣,1</sup> ·
[Simone Bombari](https://scholar.google.com/citations?user=IrWxywgAAAAJ)<sup>✣,2</sup> ·
[Tatiana Tommasi](https://scholar.google.com/citations?user=ykFtI-QAAAAJ)<sup>1</sup> ·
[Marco Mondelli](https://scholar.google.com/citations?user=BHdSb5AAAAAJ)<sup>2</sup>**

<sup>1</sup> Politecnico di Torino, Italy &nbsp;&nbsp; 
<sup>2</sup> Institute of Science and Technology Austria (ISTA)

<sup>✣</sup> Joint first-authorship, equal contribution.

**ICLR 2026**

</div>

---

<div align="center">
<img width="1956" height="556" alt="image" src="https://github.com/user-attachments/assets/a5dcf0dc-32cb-4296-a081-0a0abebef405" />
</div>

_Large-scale deep learning models are known to memorize parts of the training set. In machine learning theory, memorization is often framed as interpolation or label fitting, and classical results show that this can be achieved when the number of parameters_ $p$ _in the model is larger than the number of training samples_ $n$ _. In this work, we consider memorization from the perspective of data reconstruction, demonstrating that this can be achieved when_ $p$ _is larger than_ $dn$_, where_ $d$ _is the dimensionality of the data. More specifically, we show that, in the random features model, when_ $p \gg dn$_, the subspace spanned by the training samples in feature space gives sufficient information to identify the individual samples in input space. Our analysis suggests an optimization method to reconstruct the dataset from the model parameters, and we demonstrate that this method performs well on various architectures (random features, two-layer fully-connected and deep residual networks). Our results reveal a law of data reconstruction, according to which the entire training dataset can be recovered as_ $p$ _exceeds the threshold_ $dn$_._

---

## Dependencies
To install the required dependencies run:
```bash
pip install -r requirements.txt
```

**NOTE:** a CUDA-capable GPU and CUDA-enabled PyTorch are expected to run the experiments.

## Datasets
By running the scripts you will automatically download CIFAR-10 from the web, if internet connection is available. Otherwise, place it inside the `datasets/` folder.

## Running The Experiments
You can find in the `launch_scripts/` folder the example scripts used to run the experiments. Please, refer to the specific scripts of each experiment for the full list of command line arguments.


## Citation

If you find our work useful in your research, you can cite it as:

```bibtex
@article{iurada2025law,
  title={A Law of Data Reconstruction for Random Features (and Beyond)},
  author={Iurada, Leonardo and Bombari, Simone and Tommasi, Tatiana and Mondelli, Marco},
  journal={arXiv preprint arXiv:2509.22214},
  year={2025}
}
```
