# Encoder

To Do

# Decoder

Python module for decoding images encoded with Payload's proof of concept custom jpeg compression algorithm

## Usage

### Virtual Environment

Can skip if you know how to use virtual environments. `requirements.txt` located in `decoder` directory.

1. Install virtualenv (or virtual environment manager of choice) with 
```bash
pip install virtualenv
```

2. Create virtual environment called `.venv` with below command. Name can be whatever you want (useful if multiple environments need to be used in same project).
```bash
python -m virtualenv .venv
```

3. Activate virtual environemnt with below command for unix terminals.
```bash
source .venv/bin/activate
```

4. Double check that name of environment precedes current location in terminal. i.e. `(.venv) user@computername: /path/...`

### Requirements

Install Python dependencies with 
```bash
pip install -r decoder/requirements.txt
```

### Running Decoder

From directory containing `decoder` module (decoder directory), run `decoder` as module. Use `-h` flag for more information.
