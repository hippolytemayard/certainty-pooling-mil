# certainty-pooling-mil

## Installation 

### Library requirements

- Pyenv 2.3.9
    - Python 3.10.5    
- Poetry 1.5.1


#### Install Pyenv

```bash
curl https://pyenv.run | bash
```

Then, add pyenv bin directory in your shell configuration file `~/.bashrc` or `~/.zshrc`

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)
```

Finally reload you shell 
```bash
source ~/.bashrc
```

#### Install Poetry


```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then, add poetry's bin directory in your shell configuration file `~/.bashrc` or `~/.zshrc`

```bash
export PATH="/home/ubuntu/.local/bin:$PATH"
```

Finally reload you shell 
```bash
source ~/.bashrc
```