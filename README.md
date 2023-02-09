👾 AlgoPärt
==========

<img src="summa/figures/approximate-patterns/approximate-patterns.jpg?raw=true" width="800" 
    title="Approximate patterns of all voices in Summa">


The project AlgoPärt attemps to reconstruct compositions of [Arvo Pärt](https://en.wikipedia.org/wiki/Arvo_P%C3%A4rt) algorithmically.
Compositions in his [tintinnabuli](https://en.wikipedia.org/wiki/Tintinnabuli) style are often composed following strict rules.
By trying to reconstruct those rules, we can better understand how his compositions work.
We refer to this method of musical analysis as *analysis by synthesis*. 


Works
-----

Currently a full reconstruction has only been attempted for one work:

- *Summa* (1977)

Tintinnabulipy
---------------

The directory [`tintinnabulipy`](/tintinnabulipy/) contains some code that allows you to work with M- and T-spaces, tintinnabuli positions and tintinnabuli processes. In particular, it makes plotting them easier:

```python
from tintinnabulipy import *
from music21.chord import Chord
from music21.scale import ConcreteScale

M = MelodicSpace(MinorScale('E2'))
T = TintinnabuliSpace(Chord('E2 G2 B2'))
M.plot(M.sequence('G3', 8), 'o-')
M.grid()
```

For more explanation, have a look at [the notebook `Tutorial.ipynb`](summa/notebooks/Tutorial.ipynb).


Python setup
------------

You can find the Python version used in `.python-version` and all dependencies 
are listed in `requirements.txt`. If you use `pyenv` and `venv` to manage 
python versions and virtual environments, do the following:

```bash
# Install the right python version
pyenv install | cat .python-version

# Create a virtual environment
python -m venv env

# Activate the environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```
