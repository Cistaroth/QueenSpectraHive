import logging
root = logging.getLogger()
print(root.handlers)
hf = logging.getLogger("huggingface_hub")
print(hf.handlers, "propagate=", hf.propagate)