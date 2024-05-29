import os
huggingface_token = "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"
os.environ["HF_TOKEN"] = huggingface_token
import logging
import sys
#from Evaluation.Crows_StereoSet import evaluate_pedb

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

from torch.cuda.amp import GradScaler, autocast

import torch.nn.functional as F

import torch as T

import torch.nn as nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer


from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel
#from reference_data import dev, fair_scores,human_scores
#from reference_data import traits, groups, tplt

#from my_utils import plot_learning_curve

import os.path
import torch
from torch.nn import functional as F

import torch.nn as nn

import matplotlib.pyplot as plt


#import warnings

import matplotlib.pyplot as plt

#%matplotlib inline
import os
os.environ['OMP_NUM_THREADS'] = '1' # speed up

class SingletonType(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class LanguageModel(nn.Module, metaclass=SingletonType):
    def __init__(self, model_prefix='gpt2'):
        super(LanguageModel, self).__init__()

        
        self.short_model_name= model_prefix
        self.model_name = self.short_model_name #language_model_dict[model_prefix]
        selected_gpu = sys.argv[1]
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(dev)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_prefix)
        self.model = GPT2LMHeadModel.from_pretrained(model_prefix, output_hidden_states=True).to(self.device)
        self.d_model = self.model.config.hidden_size
        self.hidden_size = self.model.config.hidden_size
        self.num_labels = self.model.config.vocab_size
        self.type = 'p'

    def forward(self, text):

        inputs= self.tokenizer(text, return_tensors='pt')

        # Move the inputs to the correct device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Ensure the model is on the correct device
        self.model.to(self.device)
        #self.model.eval()
        outputs = self.model(**inputs)

        return outputs.hidden_states[-1]  # return embeddings , later check if you need to access specific index of it

    def forward_group(self, text, group_idx):
        device = T.device('cuda')
        inputs = self.tokenizer(text, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        #self.model.eval()
        outputs = self.model(**inputs)

        # Get the last hidden state
        #last_hidden_states = outputs.hidden_states[-1]
        last_hidden_states = outputs.hidden_states[12]  # Last layer hidden states

        if isinstance(group_idx, int):
            group_x = last_hidden_states.squeeze()[group_idx]
        elif isinstance(group_idx, list):
            group_x = T.mean(last_hidden_states.squeeze()[group_idx[0]:group_idx[-1]+1], dim=0)
        else:
            raise TypeError("group_idx must be either an integer or a list/tuple of integers.")

        return group_x

    def forward_logits(self, sentence):

        inputs = self.tokenizer(sentence, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)

        #self.model.eval()
        with T.no_grad():
            outputs = self.model(input_ids)
            #last_hidden_state = outputs.last_hidden_state

        # GPT-2 directly outputs logits as part of its forward pass
        logits = outputs.logits.squeeze(0)
        # Remove the batch dimension if necessary

        return logits

    def forward_wte(self, sentence):

        inputs = self.tokenizer(sentence, return_tensors="pt")
        
        # Get the token ids and remove the batch dimension
        input_ids = inputs['input_ids'].squeeze()  # Move input tensor to the same device as the model
        input_ids =input_ids.to(self.device)

        # Get the word token embeddings (WTE)
        with T.no_grad():
            embeddings = self.model.transformer.wte(input_ids)
        
        return embeddings

    def forward_action(self, action):
        self.model = self.model.to(self.device)
        outputs = self.model.transformer(inputs_embeds=action)

        return outputs#.hidden_states[-1].squeeze(0)

    def forward(self, model, candidate_set):
        # Number of samples for each strategy
        num_samples_iid = int(self._num_samples * self.weight)
        num_samples_thompson = self._num_samples - num_samples_iid

        # IID Normal Sampling
        iid_samples = torch.randn(num_samples_iid, candidate_set.shape[-1], device=candidate_set.device)

        # Thompson Sampling
        model.eval()
        with torch.no_grad():
            # Calculate the range (upper bound - lower bound) for each dimension
            bounds_range = self.bounds[1] - self.bounds[0]

            # Generate random points within the bounds for each dimension
            # The shape of random_points will be (num_samples_thompson, candidate_set.shape[-1])
            random_points = self.bounds[0] + bounds_range * torch.rand(num_samples_thompson, candidate_set.shape[-1], device=candidate_set.device)

            # Obtain the posterior distribution at these random points
            posterior = model.posterior(random_points)

            # Sample from the posterior distribution
            thompson_samples = posterior.sample(sample_shape=torch.Size([num_samples_thompson]))
        # Combine both sets of samples
        combined_samples = torch.cat([iid_samples, thompson_samples], dim=0)

        return combined_samples