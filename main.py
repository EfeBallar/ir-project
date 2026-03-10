# create venv
# python -m venv venv

# activate it
# source venv/bin/activate

# install dependencies
# pip install -r requirements.txt

import json
import datasets
import torch
import transformers
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM