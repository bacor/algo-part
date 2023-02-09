Algo Pärt
==========

<img src="summa/figures/approximate-patterns/approximate-patterns.jpg?raw=true" width="800" 
    title="Approximate patterns of all voices in Summa">


The project *Algo Pärt* attemps to reconstruct compositions of Arvo Pärt algorithmically.
Compositions in his tintinnabuli style are often composed following strict rules.
By trying to reconstruct those rules, we can better understand how his compositions work.
We refer to this method of musical analysis as *analysis by synthesis*. 


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
