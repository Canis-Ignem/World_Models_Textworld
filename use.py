import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from time import time
import json


class USE():
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.embed_size = 512
        
    def __len__(self):
        return 512
    
    def embed(self, text):
        enc = self.model([text])

        return enc
    
    def embed_sentences(self, l_text):
        enc = self.model(l_text)
        
        return enc


