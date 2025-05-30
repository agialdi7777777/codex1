# SMS Ad Generator

This repository contains a simple example of generating SMS advertisement text using a Hugging Face model with LangChain.

## Setup

1. Create a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies. To use an RTX2000-series GPU, install PyTorch with CUDA support before installing the rest:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   pip install -r requirements.txt
   ```
The requirements file includes `transformers`, `langchain`, and other packages used by the script.

3. Alternatively, create and activate a conda environment:
   ```bash
   conda create -n sms-ad-generator python=3.10
   conda activate sms-ad-generator
   ```
4. Install Jupyter so you can run the script in a notebook:
   ```bash
   pip install jupyter  # or: conda install notebook
   ```

## Usage

Run the `ads_generator.py` script and provide the product or topic as an argument:

```bash
python ads_generator.py "Eco-friendly water bottles"
```

The script loads a text generation model from Hugging Face, creates a prompt with LangChain, and prints a short SMS advertisement message.

## Notebook Example

Launch a notebook from the project directory:

```bash
jupyter notebook
```

In a new notebook cell you can execute the script like so:

```python
!python ads_generator.py "Eco-friendly water bottles"
```
