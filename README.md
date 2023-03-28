# Qibochem

Qibochem is a plugin to [Qibo](https://github.com/qiboteam/qibo) for chemistry applications.


## Installation

```
git clone https://github.com/qiboteam/qibochem.git
cd qibochem
pip install .
```

### Writing tutorials

Following qiboteam's [tutorial](https://github.com/qiboteam/tutorials) repository, add the following lines into your `.git/config` file.
This will clean all your notebook's output when you make a `push`.

```
[filter "jupyter_clear_output"]
    clean = "jupyter nbconvert --stdin --stdout --log-level=ERROR \
            --to notebook --ClearOutputPreprocessor.enabled=True"
    smudge = cat
    required = true

[core]
    attributesfile = .gitattributes
```

