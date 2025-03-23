test: pip install transformers==4.44.2
train:pip install transformers==4.41.2

## Sigularity Problem
RuntimeError: Failed to import transformers.trainer because of the following error (look up to see its traceback):
cannot import name 'is_mlu_available' from 'accelerate.utils'