README
=====

ALFA Activation Analysis for Deep Complex-Valued Neural Networks
================================================================

This experiment studies how the initialization parameter alpha affects
the activations of very deep complex-valued neural networks (CVNNs)
using the split-tanh activation function.

The experiment is designed especially for extremely deep networks,
such as 1,000 or 10,000 layers.

The main initialization method used in this experiment is:

    structured_preserve

The experiment evaluates several alpha values and records the
activation distributions at selected layers.


1. PURPOSE
==========

The goal is to investigate activation behavior for different values
of alpha in very deep complex-valued neural networks.

The activation function is:

    split_tanh(z) =
        tanh(Re(z)) + i tanh(Im(z))

The experiment can be used to study:

- Activation collapse
- Saturation
- Vanishing activations
- Growth or decay of activation magnitude
- Real and imaginary component distributions
- Complex-plane distributions
- Behavior at early, middle, and very deep layers


2. PROJECT LOCATION
===================

The experiment is located in:

    src/clifford_network/alfa_analysis/

Important files:

    experiment.py
        Main experiment implementation.

    plots.py
        Functions for creating activation distribution plots.

    __main__.py
        Allows the experiment to be executed as a Python module.

Results are saved by default in:

    src/clifford_network/alfa_analysis/results/


3. ENVIRONMENT
==============

Activate the project environment first:

    cd ~/clifford_network

    source /home/mad07/pytorch_env/bin/activate


4. CHECK CUDA
=============

Before running a large experiment, check that PyTorch can access
the GPU:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

You should see:

    True

followed by the GPU name.


5. SMOKE TEST
=============

Before running a large 1,000 or 10,000 layer experiment, run a small
test to make sure everything is working.

Recommended CPU smoke test:

    python -m clifford_network.alfa_analysis \
        --device cpu \
        --depth 3 \
        --hidden-size 4 \
        --num-samples 2 \
        --batch-size 2 \
        --layers 1 2 3 \
        --alfa-values 0.0085

This should finish quickly.

A slightly larger GPU smoke test is:

    python -m clifford_network.alfa_analysis \
        --device cuda \
        --depth 10 \
        --hidden-size 8 \
        --num-samples 10 \
        --batch-size 10 \
        --layers 1 5 10 \
        --alfa-values 0.0085


6. BASIC EXPERIMENT
===================

The default experiment uses:

    Depth       = 1000
    Hidden size = 32
    Samples     = 3000
    Activation  = split_tanh

Run:

    python -m clifford_network.alfa_analysis \
        --device cuda \
        --depth 1000 \
        --hidden-size 32 \
        --num-samples 3000 \
        --batch-size 3000 \
        --layers 1 10 20 250 500 750 1000 \
        --alfa-values 1e-5 1e-3 0.0085 0.1 1 10


7. 10,000-LAYER EXPERIMENT
===========================

For the main deep-network experiment, use:

    python -m clifford_network.alfa_analysis \
        --device cuda \
        --depth 10000 \
        --hidden-size 32 \
        --num-samples 3000 \
        --batch-size 3000 \
        --layers 1 10 20 2500 5000 7500 9000 10000 \
        --alfa-values 1e-5 1e-3 0.0085 0.1 1 10

This evaluates the following alpha values:

    0.00001
    0.001
    0.0085
    0.1
    1
    10


8. ONLY LOOK AT LAYER 9000
==========================

If the main goal is to inspect the activation distribution around
layer 9000 in a 10,000-layer network, use:

    python -m clifford_network.alfa_analysis \
        --device cuda \
        --depth 10000 \
        --hidden-size 32 \
        --num-samples 3000 \
        --batch-size 3000 \
        --layers 9000 \
        --alfa-values 1e-5 1e-3 0.0085 0.1 1 10

Important:

The network still contains all 10,000 layers.

The option:

    --layers 9000

only controls which activation is saved and analyzed.

Therefore, the network is still forwarded through all 10,000 layers.


9. RECOMMENDED LAYER SELECTION
==============================

For studying how activations evolve through a 10,000-layer network,
the following layers are useful:

    1
    10
    20
    2500
    5000
    7500
    9000
    10000

These provide information about:

    Early layers:
        1, 10, 20

    Middle layers:
        2500, 5000, 7500

    Very deep layers:
        9000, 10000


10. COMMAND-LINE OPTIONS
========================

The experiment supports the following main options.

--device

Select the computation device.

Examples:

    --device cuda

or:

    --device cpu


--depth

Number of network layers.

Example:

    --depth 1000

or:

    --depth 10000


--hidden-size

Number of hidden units.

Example:

    --hidden-size 32


--num-samples

Number of random complex input samples.

Example:

    --num-samples 3000


--batch-size

Number of samples processed in each forward pass.

Example:

    --batch-size 3000


--layers

Activation layers to save.

Example:

    --layers 1 10 100 1000

For a 10,000-layer network:

    --layers 1 2500 5000 7500 9000 10000


--alfa-values

Alpha values used for initialization.

Example:

    --alfa-values 1e-5 1e-3 0.0085 0.1 1 10


--activation

Activation function.

Default:

    --activation split_tanh


--bins

Number of histogram bins.

Example:

    --bins 120


--seed

Random seed.

Example:

    --seed 0


--output-dir

Custom output directory.

Example:

    --output-dir results/my_experiment


11. OUTPUT FILES
================

Results are stored in:

    src/clifford_network/alfa_analysis/results/


The main files are:

    activation_summary.csv

Contains numerical statistics for the activation distributions.

Typical statistics include:

    count
    mean
    std
    min
    p01
    p05
    p50
    p95
    p99
    max

The analysis also includes:

    real component statistics
    imaginary component statistics
    magnitude statistics
    saturation fraction
    near-zero fraction


12. ACTIVATION NPZ FILES
========================

Raw activation values are saved in:

    results/activations/

Each NPZ file contains the activation values together with metadata
such as:

    dataset
    alpha
    depth
    layer
    activation function


These files can be loaded later with NumPy without rerunning the
network.


13. COMPLEX-PLANE PLOTS
=======================

Static complex-plane plots are saved in:

    results/plots/complex_plane/


Each plot shows:

    x-axis = Real(z)
    y-axis = Imaginary(z)

The plot allows visual inspection of the distribution of complex
activations.

These plots are particularly useful for comparing different alpha
values.

For example:

    alpha = 1e-5
    alpha = 0.001
    alpha = 0.0085
    alpha = 0.1
    alpha = 1
    alpha = 10


14. ACTIVATION DISTRIBUTION REPORT
==================================

The experiment also creates an interactive HoloViews/Bokeh report:

    results/plots/activation_distributions.html

Open this file in a web browser.

The report contains distributions for:

    Real(z)
    Imag(z)
    |z|
    Complex plane


15. INTERPRETING THE RESULTS
============================

The most important quantities to examine are:

A. Mean

The average activation value.

A mean close to zero is generally expected for a balanced distribution.


B. Standard deviation

Measures the spread of the activations.

Very small standard deviation can indicate activation collapse.


C. Magnitude

The magnitude is:

    |z| = sqrt(Re(z)^2 + Im(z)^2)

This is useful for understanding whether the complex activation
remains sufficiently large through the network.


D. Saturation

For split-tanh, saturation occurs when the real or imaginary input
becomes sufficiently large.

Strong saturation can indicate that many neurons are operating in
the flat region of tanh.


E. Near-zero activations

A high near-zero fraction may indicate activation collapse or
vanishing activations.


16. COMPARING ALPHA VALUES
==========================

The main purpose of the experiment is to compare different alpha
values.

For each alpha, examine:

    1. Activation magnitude
    2. Standard deviation
    3. Saturation fraction
    4. Near-zero fraction
    5. Real distribution
    6. Imaginary distribution
    7. Complex-plane distribution

Then compare these quantities across network depth.


17. IMPORTANT FOR VERY DEEP NETWORKS
====================================

For a 10,000-layer network, it is not enough to inspect only the
first layer.

An initialization can produce reasonable activations initially but
still lead to collapse after thousands of layers.

Therefore, compare:

    Layer 1
    Layer 10
    Layer 20
    Layer 2500
    Layer 5000
    Layer 7500
    Layer 9000
    Layer 10000

This allows us to determine whether the activation distribution is
stable throughout the network.


18. REPRODUCIBILITY
===================

The experiment uses a random seed.

For example:

    --seed 0

To repeat the experiment with another seed:

    --seed 1

or:

    --seed 2

For a more robust statistical study, multiple seeds can be used.


19. RECOMMENDED MAIN EXPERIMENT
===============================

For the current study, the recommended command is:

    cd ~/clifford_network

    source /home/mad07/pytorch_env/bin/activate

    python -m clifford_network.alfa_analysis \
        --device cuda \
        --depth 10000 \
        --hidden-size 32 \
        --num-samples 3000 \
        --batch-size 3000 \
        --layers 1 10 20 2500 5000 7500 9000 10000 \
        --alfa-values 1e-5 1e-3 0.0085 0.1 1 10 \
        --seed 0


20. QUICK TEST FOR LAYER 9000
=============================

If you only want to investigate layer 9000:

    cd ~/clifford_network

    source /home/mad07/pytorch_env/bin/activate

    python -m clifford_network.alfa_analysis \
        --device cuda \
        --depth 10000 \
        --hidden-size 32 \
        --num-samples 3000 \
        --batch-size 3000 \
        --layers 9000 \
        --alfa-values 1e-5 1e-3 0.0085 0.1 1 10 \
        --seed 0


21. RESULT STRUCTURE
====================

After a successful run, the directory should look approximately like:

    results/
    |
    |-- activation_summary.csv
    |
    |-- metadata.json
    |
    |-- activations/
    |     |-- *.npz
    |
    |-- plots/
          |
          |-- complex_plane/
          |     |-- *.png
          |
          |-- activation_distributions.html


22. TROUBLESHOOTING
===================

If CUDA is not available, check:

    python -c "import torch; print(torch.cuda.is_available())"

If this returns:

    False

run the experiment on CPU for testing:

    --device cpu


If GPU memory becomes a problem, reduce:

    --num-samples

or:

    --batch-size

For example:

    --num-samples 1000
    --batch-size 100


For a quick test, reduce the depth:

    --depth 10


23. IMPORTANT NOTE ABOUT LAYER NUMBERS
======================================

The --layers argument uses 1-based activation-layer numbering.

For example:

    --layers 1

means the first activation layer.

    --layers 9000

means the 9000th activation layer.

Internally, the corresponding module name is zero-based, so layer
9000 corresponds to:

    activation_8999


24. SCIENTIFIC GOAL
===================

The central scientific question is whether the initialization
parameter alpha can maintain a useful activation distribution in
extremely deep complex-valued neural networks using split-tanh.

In particular, we want to determine whether some alpha values lead
to:

    - stable activation variance
    - reduced saturation
    - reduced activation collapse
    - stable complex magnitude
    - better propagation through 10,000 layers

The results can then be used to motivate and analyze robust
initialization strategies for very deep CVNNs.


END OF README
============= 