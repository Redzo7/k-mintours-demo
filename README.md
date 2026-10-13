# Drone Multidepot Segment Covering - Demostrator 
## Description
This software has been developed for the Computer Networks and Distributed Systems university course on ELTE Computer Science MSc, for the 2026/27/1 semester.

The software provides a demonstrator and a statistical software for the results of the paper called [An efficient algorithm for a drone multidepot 2 segment covering problem](https://perso.limos.fr/~lulorieau/docs/papers/Drone.pdf) by Laurent Beaudou et al.

## Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda)
- Python 3.13 (installed automatically into the conda environment)

| Package      | Purpose                                              |
|--------------|------------------------------------------------------|
| numpy        | geometry, closed-form maximal-tour lengths           |
| scipy        | root finding for equilibrium points (`brentq`)       |
| matplotlib   | drawing instances, tours and algorithm steps         |
| pandas       | collecting experiment results                        |
| seaborn      | statistical plots                                    |
| jupyterlab, ipywidgets | interactive demo notebooks                 |
| pytest, ruff | tests and linting                                    |

*All packages are installed from the `conda-forge` channel, which provides
Python 3.13 builds of each.*

## Installation guide
1. **Clone the repository**

```bash
    git clone https://github.com/Redzo7/k-mintours-demo
    cd k-mintours-demo
```

2. **Create the conda environment** from `environment.yml`

```bash
    conda env create -f environment.yml
```

3. **Activate the environment**

```bash
    conda activate drone-cover
```

4. **Check the installation**

```bash
   python --version          # should print Python 3.13.x
   python -c "import numpy, scipy, matplotlib, pandas; print('OK')"
```

## Project structure

```
drone-cover/
├── environment.yml
├── README.md
├── src/drone_cover/
│   ├── geometry.py      # reachability, f/g maximal-tour functions
│   ├── dominance.py     # dominated-depot removal, depot ordering
│   ├── equilibrium.py   # equilibrium points between depots
│   ├── greedy.py        # 1-depot greedy algorithm (Bereg et al.)
│   ├── kmintours.py     # recursive k-depot algorithm
│   ├── generator.py     # random instance generation
│   └── demo.py          # step-by-step visualisation
├── notebooks/
└── tests/
```

## References
 
- Beaudou, Foucaud, Lorieau, Misra — *An efficient algorithm for a drone
  multidepot segment covering problem.*
- Bereg, Díaz-Báñez, Kasiuk, Pérez-Cutiño, Rodríguez — *Covering segments on a
  line with drones.* Information Processing Letters 188:106540.