# Simple Input Convex Neural Network Implementation

This repository contains a simple implementation of an
Input Convex Neural Network as described in:

Amos, Brandon, Lei Xu, and J. Zico Kolter. 
"Input convex neural networks." 
International Conference on Machine Learning. 
PMLR, 2017.

This a prototype implementation of that idea using skip connections
and constraining weights to be non-negative (or negative in last 
layer to create a concave network with respect to inputs) using 
torch.clamp(). As for now, It is to be trained normally with ADAM 
stochastic gradient descent without any special concern to the weight
space constraints, though it may be possible to create a better 
optimizer in the future.

## Running the code

This repository is being developed in a Linux environment with 
Anaconda, python 3.8, and PyTorch. The repository contains a .yml 
file indicating the particulars of the working environment.

The following code should create a similar environment:

```
conda env create --file python-3-8.yml
```

Activate the new environment with:

```
conda activate python-3-8
```

Run the code by calling while in the appropriate directory:

```
python3 main.py
```

This above call should train an ICNN and report test set feedback.