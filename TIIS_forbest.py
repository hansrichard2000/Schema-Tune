import os
huggingface_token = "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"
os.environ["HF_TOKEN"] = huggingface_token
import copy
import math
import logging
import sys
from eval_stereoset import eval_stereoset 
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
from pedb_main.evaluate import evaluate_pedb
from torch.cuda.amp import GradScaler, autocast

import re
import gc
from functools import partial
#from google.colab import drive
import torch.nn.functional as F

import torch as T
import pdb
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from scipy.optimize import minimize
#from torch.nn.functional import cosine_similarity
from scipy.optimize import minimize
from itertools import combinations
#from gpt_critic_minimal import CriticNetwork
from typing import List
import time
from itertools import combinations
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pdb
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import botorch
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
#from reference_data import dev, fair_scores,human_scores
#from reference_data import traits, groups, tplt

#from my_utils import plot_learning_curve

import json
import os.path
import pathlib
#import pyperclip
from collections import OrderedDict


from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os
import random
import torch.nn as nn
from collections import defaultdict
import torch.optim as optim
from collections import defaultdict
import matplotlib.pyplot as plt
#from botorch.models import KroneckerMultiTaskGP
from botorch.models.higher_order_gp import HigherOrderGP
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.utils.transforms import standardize, normalize
import torch.nn.functional as F
from botorch.sampling import MCSampler
import gpytorch
#from gpytorch.models import MultitaskGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal

from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import RBFKernel, MultitaskKernel
from gpytorch.models import ExactGP
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors import GPyTorchPosterior


from botorch.fit import fit_gpytorch_model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize

from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement

from botorch.utils.multi_objective import infer_reference_point
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning



from botorch.models.multitask import MultiTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize


from gpytorch.likelihoods import MultitaskGaussianLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qKnowledgeGradient
from botorch.optim import optimize_acqf
from gpytorch.likelihoods import GaussianLikelihood
from botorch.utils.transforms import standardize, normalize

from botorch.models import SingleTaskGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound



#import warnings
from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement, qSimpleRegret
from botorch.acquisition.risk_measures import VaR
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples, draw_sobol_normal_samples
from botorch.utils.transforms import unnormalize
from botorch.test_functions import SixHumpCamel
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from botorch.models.transforms.input import Normalize


import torch
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
#%matplotlib inline



from argparse import Namespace
import os
os.environ['OMP_NUM_THREADS'] = '1' # speed up
import numpy as np
from DGEMO.problems.common import build_problem
from DGEMO.mobo.algorithms import get_algorithm
from DGEMO.visualization.data_export import DataExport
from DGEMO.arguments import get_args
from DGEMO.utils import save_args, setup_logger
import gc
import torch







#warnings.filterwarnings("ignore")



# BERT (e.g., "bert-base-uncased")
# GPT-2 (e.g., "gpt2")
# RoBERTa (e.g., "roberta-base")
# XLNet (e.g., "xlnet-base-cased")
# DistilBERT (e.g., "distilbert-base-uncased")
# ALBERT (e.g., "albert-base-v2")
# T5 (e.g., "t5-small")
# GPT-3 (as of my training cut-off in September 2021, GPT-3 wasn't directly available in the Transformers library, but its architecture is similar to GPT-2)
# DeBERTa (e.g., "microsoft/deberta-base")
# ELECTRA (e.g., "google/electra-small-discriminator")


# sampling_group_pairs= [ ('man', 'woman'),('stepfather', 'stepmother')]


def print_gpu_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'Total GPU Memory: {t}, Reserved: {r}, Allocated: {a}, Free: {f}')


class SingletonType(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]



class LanguageModel(nn.Module, metaclass=SingletonType):
    def __init__(self, model_prefix='gpt-2'):
        super(LanguageModel, self).__init__()

        language_model_dict = {
            'bert': 'bert-base-uncased',
            'roberta': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            'albert': 'albert-base-v2',
            'xlnet': 'xlnet-base-cased',
            'electra': 'electra-base',
            't5': 't5-small',
            'gpt-2': 'gpt2',
            'gpt-3': 'gpt2',  # Replace with appropriate GPT-3 model name if available
            'deberta': 'bert-base-uncased'  # Replace with appropriate DeBERTa model name if available
        }
        # if model_prefix not in language_model_dict:
        #     raise ValueError(f"Model prefix '{model_prefix}' is not supported.")
        #self.short_model_name= model_prefix
        self.short_model_name= 'gpt2'
        self.model_name =self.short_model_name #language_model_dict[model_prefix]
        selected_gpu = sys.argv[1]

        self.device = torch.device(f'cuda:{selected_gpu}')

        #self.device = T.device('cuda')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True).to(self.device)
        self.d_model=  self.model.config.hidden_size
        self.hidden_size=self.model.config.hidden_size
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
        #print (inputs, sentence, "bye")
        # Get the token ids and remove the batch dimension
        input_ids = inputs['input_ids'].squeeze()  # Move input tensor to the same device as the model
        input_ids =input_ids.to(self.device)

        # Get the word token embeddings (WTE)
        with T.no_grad():
            embeddings = self.model.transformer.wte(input_ids)
        #print ("PLM embeddings.shape",embeddings.shape)
        return embeddings

    def forward_action(self, action):
        self.model = self.model.to(self.device)

        #print("forward_action")
        #print("Action shape:", action.shape)  # Debugging line

        outputs = self.model.transformer(inputs_embeds=action)

        #print ("outputs",outputs)
        return outputs#.hidden_states[-1].squeeze(0)


# class MixedSampler(MCSampler):

#     def __init__(self, bounds, weight=0.2, sample_shape=torch.Size([]), **kwargs):
#         # Ensure sample_shape is passed to the superclass constructor
#         super().__init__(sample_shape=sample_shape, **kwargs)
#         self.bounds = bounds
#         self.weight = weight

# class MixedSampler(MCSampler):
#     def __init__(self, bounds, num_samples, weight=0.2, sample_shape=torch.Size([]), **kwargs):
#         # Ensure sample_shape is passed to the superclass constructor
#         super().__init__(num_samples=num_samples, sample_shape=sample_shape, **kwargs)
#         self.bounds = bounds
#         self.weight = weight
#         # Additional initialization as needed

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

# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ConstantMean(), num_tasks=2
#         )
#         self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# class MultiOutputGPModel(MultitaskGPModel):
#     def __init__(self, train_x, train_y, num_tasks):
#         # Define a multitask likelihood
#         likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
#         super().__init__(train_x, train_y, likelihood)

#         # Multitask kernel

#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=num_tasks, rank=1)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



# class MultiOutputGPModel(ExactGP):
#     def __init__(self, train_x, train_y, num_tasks):
#         # Initialize the likelihood

#         likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
#         super().__init__(train_x, train_y, likelihood)

#         # Use MultitaskMean with a ConstantMean base mean function for each task
#         self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)

#         # Multitask kernel with RBF base kernel
#         self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=num_tasks, rank=1)
#         self.num_outputs = num_tasks  # Assuming num_tasks corresponds to the number of outputs
#         self._train_data_shape = train_x.shape[:-1]  # Exclude feature dimension

#     @property
#     def batch_shape(self):
#         # Return the shape of the batch dimensions
#         # This example assumes train_x was of shape [batch_size, n, d]
#         # Adjust according to how your model handles batches
#         return self._train_data_shape[:-1]  # Exclude the last dimension (n)

#     def forward(self, x):
#         # The mean_module now directly returns a [batch_size, num_tasks] tensor
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

#     def posterior(self, X, observation_noise=False, **kwargs):
#         self.eval()  # Set the model to evaluation mode
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             mvn = self(X)
#             if observation_noise:
#                 mvn = self.likelihood(mvn, X)
#         return GPyTorchPosterior(mvn)


# # Usage example with an acquisition function
# # Assuming `model` is your trained model and `candidate_set` is your set of candidate points
# # acq_func = qMaxValueEntropy(model, candidate_set, posterior_transform=SelectFirstOutputTransform())

# from botorch.posteriors import GPyTorchPosterior
# from botorch.acquisition.objective import PosteriorTransform
# from torch import Tensor

# class SelectFirstOutputTransform(PosteriorTransform):
#     def evaluate(self, Y: Tensor) -> Tensor:
#         # This method is required by the abstract base class but may not be
#         # necessary for our specific use case. Implementing with a pass-through
#         # or raising an error if called directly.
#         raise NotImplementedError("This transform does not implement evaluate directly.")

#     def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
#         """
#         Selects the first output from a multi-output posterior.

#         Args:
#             posterior: The multi-output posterior distribution from a model.

#         Returns:
#             A new GPyTorchPosterior object corresponding to the first output.
#         """
#         # Check if the posterior is a GPyTorchPosterior with multi-output
#         if not isinstance(posterior, GPyTorchPosterior):
#             raise ValueError("Expected a GPyTorchPosterior object.")

#         # Extract the mean and covariance of the first output
#         new_mean = posterior.mean[..., 0]
#         if posterior.variance.ndim > 1:
#             new_covar = posterior.mvn.lazy_covariance_matrix[..., 0:1, 0:1]
#         else:
#             new_covar = posterior.variance[..., 0]

#         # Construct a new posterior object for the first output
#         new_posterior = GPyTorchPosterior(
#             mvn=posterior.mvn.__class__(mean=new_mean, covariance_matrix=new_covar)
#         )
#         return new_posterior




class SparseGPModel(ExactGP, GPyTorchModel):
    _num_outputs = 1  # To inform BoTorch of the model's output dimension
    
    def __init__(self, train_X, train_Y, likelihood, inducing_points):
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.inducing_points = inducing_points
        self.covar_module.base_kernel.lengthscale = 1.0
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class Noise:
    def __init__(self, actor, PLM, reward_obj, sampling_group_pairs, device, batch_size,lambda1, variance_scale_factor=0.5, template= "the {} is powerful."
                 ,num_samples=500, maxiter=32, improvement_threshold=0.0001,mixed_sampling_weight=0.2,num_restarts=2,
                    raw_samples=500,batch_limit=8):
        self.lambda1=lambda1
        self.num_tokens=2
        self.num_samples=num_samples
        self.maxiter=maxiter
        self.improvement_threshold=improvement_threshold
        self.mixed_sampling_weight=mixed_sampling_weight
        self.num_restarts=num_restarts
        self.raw_samples=raw_samples
        self.batch_limit=batch_limit

        #self.mu_size = mu.size()
        #self.sigma = sigma
        # self.dt = dt
        # self.x0 = x0
        #self.x_prev=mu
        self.variance_scale_factor=variance_scale_factor #0.1
        self.traits=traits
        self.batch_size= batch_size
        self.tokenizer= actor.tokenizer
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device =device# T.device('cuda')

        self.embedding_dim= actor.input_dim
        self.sampling_group_pairs= sampling_group_pairs # [('man', 'woman'), ('stepfather', 'stepmother')]
        self.num_groups= len(self.sampling_group_pairs)
        self.num_groups_per_pair = len(self.sampling_group_pairs[0])  # Assuming each tuple in group_pairs has the same number of elements
        self.template = template#"the {} is powerful."
        self.defining_group_sets = [('man', 'woman'), ('he', 'she'), ('father', 'mother'), ('boy', 'girl'),
            ('stepfather','stepmother'),
            ('gentlemen','ladies'),
            ('male', 'female'),
            ('brother', 'sister'),
            ('grandfather','grandmother'),
            ('groom', 'bride'),
            ('husband','wife'),
            ('gentleman', 'lady'),
            ('son', 'daughter'),
            ('boyfriend', 'girlfriend'),
            ('daddy', 'mommy')]
            #('sir, ma'am ')
        self.defining_group_sets_1= [ ('man', 'woman'), ('he', 'she'), ('father', 'mother'), ('boy', 'girl'),('groom', 'bride'),

            ('brother', 'sister')]
        #self.U, self.S= self.bias_subspace(actor)
        self.U= self.bias_subspace(actor)

        # self.variances= self.get_variances()
        self.bias_subspace_dim= len(self.sampling_group_pairs)
        #self.bounds=self.set_bounds()
        self.samples=[]
        self.default_values= self.initialize_data()

        self.start_cond=True
        self.mu_prime_Batch=torch.zeros((self.batch_size, self.num_groups_per_pair, self.num_groups,self.num_tokens, self.embedding_dim))
        self.reward= reward_obj
        self.device_cpu = torch.device("cpu")
        self.model=None

# Function to get GPT-2 static embeddings
    def get_gpt2_static_embedding(self, word, actor):
        # Generate the sentence using the template
        sentence = self.template.format(word)

        # Tokenize the sentence
        tokenized_input = actor.tokenizer(sentence, return_tensors="pt")
        input_ids = tokenized_input['input_ids'].squeeze().to(self.device)



        # Find the indices corresponding to the word
        word_token = actor.tokenizer.tokenize('Ġ' + word)
        word_ids = actor.tokenizer.convert_tokens_to_ids(word_token)
        word_indices = [i for i, token_id in enumerate(input_ids.tolist()) if token_id in word_ids]

        # Calculate the sentence embeddings
        with torch.no_grad():  # Ensure no gradients are calculated
          sentence_embedding = actor.model.transformer.wte(input_ids)

        # Extract the embedding for the word
        word_embedding = sentence_embedding[word_indices, :].mean(dim=0)

        return word_embedding


    def find_elbow_point(singular_values, plot_filename='singular_values_plot.png'):
    # Ensure singular_values is a 1D numpy array
        if isinstance(singular_values, torch.Tensor):
            singular_values = singular_values.cpu().detach().numpy()
        # if singular_values.ndim != 1:
        #     raise ValueError("singular_values must be a 1D array or tensor.")

        # Plot the singular values
        print ("singular_values",singular_values)
        plt.plot(singular_values)
        plt.xlabel('Number of Components')
        plt.ylabel('Singular Value')
        plt.title('SVD Singular Values')
        plt.savefig(plot_filename, format='png', dpi=300)
        plt.close()

        # Calculate the rate of change of the singular values
        ratios = np.diff(singular_values) / singular_values[:-1]
        elbow_point = np.argmin(ratios) + 1  # Adding 1 because np.diff reduces the length by 1
        return elbow_point
    # this needs to be modified in later developments to handle 3d tensors.
    # def bias_subspace(self, actor):
    #     # Construct the matrix C
    #     C = torch.zeros((768, 768))  # GPT-2 embedding size is 768
    #     C = C.to(self.device)
    #     for pair in self.defining_group_sets_1:
    #         embedding1 = self.get_gpt2_static_embedding(pair[0], actor)  # Ensure this returns a PyTorch tensor
    #         embedding2 = self.get_gpt2_static_embedding(pair[1], actor)  # Ensure this returns a PyTorch tensor
    #         mean_embedding = (embedding1 + embedding2) / 2
    #         diff1 = embedding1 - mean_embedding
    #         diff2 = embedding2 - mean_embedding
    #         device = actor.device  # or directly use 'cuda:0' or 'cpu'

    #         # Your existing code to calculate embeddings...

    #         # Ensure tensors are on the same device before the operation
    #         diff1 = diff1.to(self.device)
    #         diff2 = diff2.to(self.device)

    #         C += torch.outer(diff1, diff1)
    #         C += torch.outer(diff2, diff2)

    #     # Perform SVD
    #     reg = 1e-3  # Try a larger regularization parameter
    #     C_reg = C + reg * torch.eye(C.size(0), device=C.device)
    #     U, S, V = torch.linalg.svd(C_reg, full_matrices=True)
    #     #U, S, V = torch.linalg.svd(C, full_matrices=True)

    #     #U, S, V = torch.linalg.svd(C)

    #     # Find the elbow point in the singular values
    #     m = self.find_elbow_point(S)  # Ensure this function is compatible with PyTorch tensors

    #     # Bias subspace can be spanned by the first few singular vectors
    #     # Transpose U to get the correct order of dimensions
    #     bias_subspace = U[:, :m].T

    #     # Singular values
    #     singular_values = S[:m]
    #     print("singular_values", singular_values)

    #     # Returning bias directions and singular values
    #     return bias_subspace, singular_values


    def randomized_svd(self,M, n_components, device='cuda'):
        random_projection = torch.randn(M.shape[1], n_components, device=device)

        # Project the input matrix M to a lower-dimensional space
        M_projected = M @ random_projection

        # Perform QR decomposition on the projected matrix to orthonormalize it
        Q, _ = torch.linalg.qr(M_projected)

        # Project M again using Q to get a smaller matrix for SVD
        B = Q.T @ M

        # Perform SVD on the smaller matrix B
        U_hat, S_approx, Vt_approx = torch.linalg.svd(B, full_matrices=False)

        # Compute the approximate left singular vectors
        U_approx = Q @ U_hat

        return U_approx, S_approx, Vt_approx

    def bias_subspace(self, actor):
      U = torch.zeros(len(self.sampling_group_pairs), self.num_tokens, actor.model.config.n_embd)
      S = torch.zeros(len(self.sampling_group_pairs))

      for i, (word1, word2) in enumerate(self.sampling_group_pairs):
          # Tokenize and get embeddings
          tokens_1 = actor.tokenizer(" " + word1, return_tensors='pt')
          tokens_2 = actor.tokenizer(" " + word2, return_tensors='pt')

          with torch.no_grad():
              # Get the token ids and remove the batch dimension
              tokens_1 = tokens_1['input_ids'].squeeze().to(self.device)
              tokens_2 = tokens_2['input_ids'].squeeze().to(self.device)

              # Get embeddings
              embeddings_1 = actor.model.transformer.wte(tokens_1)
              embeddings_2 = actor.model.transformer.wte(tokens_2)

              # Calculate the standard deviation for the Gaussian noise based on the max values
              max_values = torch.max(embeddings_1, embeddings_2)
              noise_std = 0.05 * torch.max(max_values)  # Proportional to the max value

              # Calculate difference and normalize
              diff = embeddings_1 - embeddings_2

              # Check if any row of diff is zero
              if torch.any(torch.all(diff == 0, dim=-1)):
                  # Find rows where diff is not zero
                  nonzero_rows = ~torch.all(diff == 0, dim=-1)
                  zero_rows = torch.all(diff == 0, dim=-1)

                  # Calculate the average norm of nonzero rows
                  avg_norm = torch.mean(torch.norm(diff[nonzero_rows], dim=-1))

                  # Determine which embeddings to normalize
                  #if torch.all(diff == 0, dim=-1)[nonzero_rows]:
                  diff[zero_rows] = avg_norm * torch.ones_like(embeddings_1[zero_rows])

              # Generate Gaussian noise with mean 0 and specified standard deviation for each dimension
              gaussian_noise = torch.randn_like(diff) * noise_std.unsqueeze(-1)
              #gaussian_noise = torch.ones_like(diff) * max_values.unsqueeze(-1)
              # Add noise in the direction of normalized difference

              norm_diff = diff / torch.norm(diff, dim=-1, keepdim=True)
              perturbed_embeddings = norm_diff #+ gaussian_noise


              # Store perturbed embeddings
              U[i] = perturbed_embeddings.unsqueeze(0)

      print("this is U", U)
      return U

    # def bias_subspace(self, actor):

    #     U = torch.zeros(len(self.sampling_group_pairs), self.num_tokens, actor.model.config.n_embd)
    #     S = torch.zeros(len(self.sampling_group_pairs))

    #     for i, (word1, word2) in enumerate(self.sampling_group_pairs):
    #         # Tokenize and get embeddings
    #         tokens_1 = actor.tokenizer(" " + word1, return_tensors='pt')
    #         tokens_2 = actor.tokenizer(" " + word2, return_tensors='pt')

    #         with torch.no_grad():
    #             # Get the token ids and remove the batch dimension
    #             tokens_1 = tokens_1['input_ids'].squeeze().to(self.device)
    #             tokens_2 = tokens_2['input_ids'].squeeze().to(self.device)

    #             # Get embeddings
    #             embeddings_1 = actor.model.transformer.wte(tokens_1)
    #             embeddings_2 = actor.model.transformer.wte(tokens_2)

    #             max_values = torch.max(embeddings_1, embeddings_2)
    #             # Calculate the standard deviation for the Gaussian noise based on the max values
    #             noise_std = 0.05 * torch.max(max_values)  # Proportional to the max value

    #             # Calculate difference and normalize
    #             diff = embeddings_1 - embeddings_2

    #             norm_diff = diff / torch.norm(diff, dim=-1, keepdim=True)

    #             # Generate Gaussian noise with mean 0 and specified standard deviation for each dimension
    #             gaussian_noise = torch.randn_like(diff) * noise_std.unsqueeze(-1)

    #             # Add noise in the direction of normalized difference
    #             perturbed_embeddings = norm_diff + gaussian_noise

    #             # Store perturbed embeddings
    #             U[i] = perturbed_embeddings.unsqueeze(0)
    #             #S[i] = torch.max(torch.norm(diff, dim=-1))  # Calculate the norm along the embedding dimension

    #             #
    #             #pdb.set_trace()

    #     print ("this is U",U )
    #     return U
    #     # U = torch.zeros(len(self.sampling_group_pairs), self.num_tokens, actor.model.config.n_embd)
        # S = torch.zeros(len(self.sampling_group_pairs))

        # for i, (word1, word2) in enumerate(self.sampling_group_pairs):
        #     # Tokenize and get embeddings
        #     tokens_1 = actor.tokenizer(" "+word1, return_tensors='pt')
        #     tokens_2 = actor.tokenizer(" "+word2, return_tensors='pt')
        #     print(tokens_1,tokens_2, "tokens_1,tokens_2" )
        #     with torch.no_grad():
        # #print (inputs, sentence, "bye")
        # # Get the token ids and remove the batch dimension
        #       tokens_1 = tokens_1['input_ids'].squeeze()  # Move input tensor to the same device as the model
        #       input_ids1 =tokens_1.to(self.device)
        #       embeddings_1 = actor.model.transformer.wte(input_ids1)
        #       #print ("embeddings1", embeddings_1)

        #       tokens_2 = tokens_2['input_ids'].squeeze()  # Move input tensor to the same device as the model
        #       input_ids2 =tokens_2.to(self.device)
        #       embeddings_2 = actor.model.transformer.wte(input_ids2)
        #       #print ("embeddings2", embeddings_2)
        #     # Extract embeddings for max_num_tokens

        #     # Calculate difference and variance
        #     diff = embeddings_1 - embeddings_2

        #     normal_additive_noise = torch.randn_like(diff)*0.05* torch.norm(diff, dim=-1,  keepdim=True)
        #     if diff.size(0) < self.num_tokens:
        #       diff.unsqueeze(0)

        #       padding = torch.zeros_like(diff)


        #       # Concatenate the original tensor with the padding to expand it
        #       # Use unsqueeze to add a dimension to each tensor, making them 2D, then concatenate along dim=0
        #       expanded_diff = torch.cat(( padding.unsqueeze(0),diff+ diff*normal_additive_noise), dim=0)
        #       #expanded_diff = torch.cat(( padding.unsqueeze(0),diff), dim=0)

        #     else:
        #       expanded_diff = diff+ normal_additive_noise*diff
        #       #expanded_diff = diff

        #     U[i] = expanded_diff.unsqueeze(0)
        #     #.squeeze(0)#.mean(dim=0)*0.5
        #     S[i] = torch.max(torch.norm(diff, dim=-1))  # Calculate the norm along the embedding dimension
        #     print ("U[i], S[i]",U[i], S[i])

        # return U, S


    # def bias_subspace(self, actor, regularization_strength=1e-3):
    #     """
    #     Calculate the bias subspace using randomized SVD.
    #     """
    #     dim= 768
    #     C = torch.zeros((768, 768), device=self.device)  # GPT-2 embedding size is 768
    #     for pair in self.defining_group_sets_1:
    #         embedding1 = self.get_gpt2_static_embedding(pair[0], actor).to(self.device)
    #         embedding2 = self.get_gpt2_static_embedding(pair[1], actor).to(self.device)
    #         mean_embedding = (embedding1 + embedding2) / 2
    #         diff1 = embedding1 - mean_embedding
    #         diff2 = embedding2 - mean_embedding

    #         C += torch.outer(diff1, diff1)
    #         C += torch.outer(diff2, diff2)

    #       # Regularization


    #     # reg_matrix = C + regularization_strength * torch.eye(C.size(0), device=self.device)

    #     # # Perform eigen decomposition
    #     # eigenvalues, eigenvectors = torch.linalg.eigh(reg_matrix)

    #     # # Ensure eigenvalues are in descending order
    #     # idx = eigenvalues.argsort(descending=True)
    #     # eigenvalues = eigenvalues[idx]
    #     # eigenvectors = eigenvectors[:, idx]

    #     # # Use eigenvalues and eigenvectors as proxies for SVD's singular values and vectors
    #     # S_approx = torch.sqrt(eigenvalues.clamp(min=0))  # Ensure non-negative
    #     # U_approx = eigenvectors
    #     # V_approx = U_approx  # For symmetric matrices, U and V are the same

    #     n_components = 2#self.find_elbow_point(C)  # Adjust this method to return an integer

    #   # return U_approx, S_approx, V_approx
    #     S_synthetic = torch.linspace(start=2, end=1, steps=n_components, device=device)

    #     # Generate random orthonormal matrices for U and V
    #     U_synthetic = torch.randn(dim, n_components, device=device)
    #     U_synthetic, _ = torch.linalg.qr(U_synthetic)  # Orthonormalize U

    #     V_synthetic = torch.randn(dim, n_components, device=device)
    #     V_synthetic, _ = torch.linalg.qr(V_synthetic)




        # # reg = 1e-2
        # # C_reg = C + reg * torch.eye(C.size(0), device=self.device)

        # # # Number of components to keep

        # # # Perform randomized SVD
        # # U, S, _ = self.randomized_svd(C_reg, n_components=n_components,  device=self.device)

        # # Bias subspace can be spanned by the first few singular vectors
        # bias_subspace = U_synthetic[:, :n_components].T

        # # Singular values
        # singular_values = S_synthetic[:n_components]
        # print("Singular values:", singular_values)

        # return bias_subspace, singular_values


    # def gaussian_noise(self, size):
    #         #return T.normal(mean=T.tensor(self.mu, device=self.device), std=self.sigma, size=(size,))
    #         mu= T.zeros(size)
    #         return T.normal(mean=mu, std=self.sigma).to(self.device)


    # check its functionality!!!

    def get_variances(self):

        variances = self.S[:]  # Variance along each principal component
        variances_list = variances.tolist()  # Convert tensor to a list of numbers
        return variances_list


    def set_bounds(self):
        bounds = []
        print ("self.variances",self.variances)
        for i, var in enumerate(self.variances):
            std_scale = np.sqrt(var) * 0.3 #self.variance_scale_factor
            #default_val = self.default_values[i]

            # Bounds for the mean value
            mean_lower_bound =  - std_scale
            mean_upper_bound =  std_scale
            bounds.append((mean_lower_bound, mean_upper_bound))
        #print(bounds, "bounds")
        return torch.tensor(bounds, requires_grad=True).T



    # def gaussian_noise_subspace_batch(self, init_x):
    #     # Convert bias_subspace from numpy array to PyTorch tensor
    #     bias_subspace_tensor = self.U
    #     #noise_mean_val.size = (self.num_bias_subspace, self.num_groups, self.num_groups_per_pair)


    #     # Iterate over each element in the batch, group_pair_id, and group_id
    #     # Initialize the noise tensor with the specified dimensions
    #     noise_tensor = torch.zeros((self.batch_size, self.num_groups_per_pair, self.num_groups, self.embedding_dim), device=self.device)

    #     for b in range(self.batch_size):
    #         for g_pair in range(self.num_groups_per_pair):
    #             for g in range(self.num_groups):
    #                 # Initialize a temporary tensor to accumulate the noise vectors
    #                 temp_noise = torch.zeros((self.embedding_dim,), device=self.device)

    #                 # Iterate over each bias subspace dimension
    #                 for i in range(self.bias_subspace_dim):
    #                     # Extract the noise value for this bias subspace dimension
    #                     noise_val = init_x[b, i, g_pair, g]

    #                     # Multiply each value in the bias subspace by the corresponding noise value
    #                     temp_noise += bias_subspace_tensor[i] * noise_val

    #                 # Assign the accumulated noise to the corresponding position in the noise tensor
    #                 noise_tensor[b, g_pair, g, :] = temp_noise

    #     return noise_tensor


    def gaussian_noise_subspace(self, init_x):
        # Convert bias_subspace from numpy array to PyTorch tensor
        bias_subspace_tensor = self.U.to(self.device)
        #print ("bias_subspace_tensor", bias_subspace_tensor.size(), bias_subspace_tensor)
        #print (init_x.size(), init_x)
        #noise_mean_val.size = (self.num_bias_subspace, self.num_groups, self.num_groups_per_pair
        # Initialize the noise tensor with the specified dimensions
        noise_tensor = torch.zeros((self.batch_size, self.num_groups,self.num_groups_per_pair,  self.num_tokens, self.embedding_dim), device=self.device)
        print ("init_x.size(0)",init_x.size())
        # Iterate over each element in the batch, group_pair_id, and group_id
        for b in range(init_x.size(0)):
            for g_pair in range(self.num_groups_per_pair):
                for g in range(self.num_groups):
                    # Initialize a temporary tensor to accumulate the noise vectors
                    temp_noise = torch.zeros((self.num_tokens, self.embedding_dim), device=self.device)

                    # Iterate over each bias subspace dimension
                    for i in range(self.bias_subspace_dim):
                        # Extract the noise value for this bias subspace dimension
                        noise_val = init_x[b, i, g_pair, g]
                        # noise = torch.randn_like(bias_subspace_tensor[i])

                        # Add the noise vector to bias_subspace_tensor[i]
                        # noisy_vector = bias_subspace_tensor[i] + noise$
                        # expanded_bias_subspace_tensor = noise.unsqueeze(0).repeat(self.num_tokens, 1)$??repeat?
                        #expanded_bias_subspace_tensor = noisy_vector.unsqueeze(0).repeat(self.num_tokens, 1)
                        #print ("bias_subspace_tensor[i].size()",bias_subspace_tensor[i].size())
                        if bias_subspace_tensor[i].size(0) == 1:
                            # Create a padding tensor of the same size with zeros (or any value you want)
                            padding = torch.zeros_like(bias_subspace_tensor[i])

                            # Concatenate the original tensor with the padding to expand it
                            # Use unsqueeze to add a dimension to each tensor, making them 2D, then concatenate along dim=0
                            expanded_tensor = torch.cat(( padding.unsqueeze(0), bias_subspace_tensor[i].unsqueeze(0)), dim=0)
                        else:
                            expanded_tensor= bias_subspace_tensor[i]

                        if g_pair == 0:
                          expanded_bias_subspace_tensor = -expanded_tensor
                        if g_pair == 1:
                          expanded_bias_subspace_tensor = expanded_tensor
                        # Multiply the expanded tensor by noise_val
                        result_tensor = expanded_bias_subspace_tensor * noise_val
                        # Multiply each value in the bias subspace by the corresponding noise value
                        temp_noise -= result_tensor
                    #print ("b, i, g_pair, g", b, g_pair, g, init_x.size(0) )
                    # Assign the accumulated noise to the corresponding position in the noise tensor
                    noise_tensor[b, g,g_pair, :, :] = temp_noise

        del bias_subspace_tensor

        return noise_tensor


    # def problem (self, init_x, LM, PLM ):

    #     noise_tensor=self.gaussian_noise_subspace( init_x)

    #     # later modify the way noisy embedding is generated.
    #     mu_prime_B=  self.calculate_noisy_embeddings(self.group_embeddings, noise_tensor, True)

    #     #
    #     rew= self.reward.calculate_reward( mu_prime_B, LM, PLM, self.lambda1)
    #     self.mu_prime_Batch=mu_prime_B
    #     num_repeats = self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups

    #     # Repeat each element in rew to match the desired shape
    #     # Note: rew is expected to be a 1D tensor of shape (batch_size,)
    #     rew_expanded = rew.repeat_interleave(num_repeats).view(self.batch_size, num_repeats)

    #     return rew.unsqueeze(0).reshape(16, 1)

    def normalize(self, tens, bounds):
      # bounds is [2, self.bias_subspace_dim]
      #bounds=bounds.to(self.device)

      lower, upper = bounds[0], bounds[1]  # Each is [self.bias_subspace_dim]

      # Expand lower and upper to match the tensor's bias_subspace_dim and then repeat
      lower_expanded = lower.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(tens.size(0), 1, self.num_groups_per_pair, self.num_groups)
      upper_expanded = upper.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(tens.size(0), 1, self.num_groups_per_pair, self.num_groups)

      norm_tensor = (tens - lower_expanded) / (upper_expanded - lower_expanded)
      del  lower_expanded, upper_expanded
      return norm_tensor

    def unnormalize(self, tens, bounds):
      #bounds=bounds.to(self.device)
      lower, upper = bounds[0], bounds[1]  # Each is [self.bias_subspace_dim]

      lower_expanded = lower.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(tens.size(0), 1, self.num_groups_per_pair, self.num_groups)
      upper_expanded = upper.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(tens.size(0), 1, self.num_groups_per_pair, self.num_groups)

      unnorm_tensor = tens * (upper_expanded - lower_expanded) + lower_expanded
      del lower_expanded, upper_expanded

      return unnorm_tensor



    # def problem(self, X, actor, PLM):
    #   # X is expected to be of shape [num_candidates, num_features]
    #   # Reshape X to the expected input shape for gaussian_noise_subspace
    #   X_reshaped = X.view(-1, self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)

    #     # Detaching the tensor from its history

    #   noise_tensor = self.gaussian_noise_subspace(X_reshaped).detach()
    #   #print (noise_tensor,"noise_tensor")
    #   # Use the detached tensor in subsequent operations

    #   mu_prime_B = self.calculate_noisy_embeddings(self.group_embeddings.detach(), noise_tensor, True).to(self.device)

    #   # Calculate the reward using the detached tensors
    #   batch_rewards = self.reward.calculate_reward(mu_prime_B.detach(), actor, PLM, self.lambda1).detach()
    #   #batch_rewards = batch_rewards.clone().detach().to(self.device).double().requires_grad_(True)
    #   cl_rew= batch_rewards.clone()
    #   #print (batch_rewards,"batch_rewards")
    #   while nan in cl_rew.item():

    #     noise_tensor = self.gaussian_noise_subspace(X_reshaped).detach()
    #     #print (noise_tensor,"noise_tensor")
    #     # Use the detached tensor in subsequent operations

    #     mu_prime_B = self.calculate_noisy_embeddings(self.group_embeddings.detach(), noise_tensor, True).to(self.device)

    #     # Calculate the reward using the detached tensors
    #     batch_rewards = self.reward.calculate_reward(mu_prime_B.detach(), actor, PLM, self.lambda1).detach()

    #   # Ensure batch_rewards is a tensor of shape [num_candidates, 1]
    #   return - batch_rewards.view(-1, 1).to(self.device)

    def calculate_mu_prime_B(self,  X, actor, PLM):

        X_reshaped = X.view(-1, self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
        #batch_rewards = torch.empty(X_reshaped.size(0), 1, device=self.device)
        noise_tensor = self.gaussian_noise_subspace(X_reshaped).detach()
        mu_prime_B = self.calculate_noisy_embeddings(self.group_embeddings.detach(), noise_tensor, True).to(self.device)

        return mu_prime_B

    def normalize_batch_mu(self, mu_prime_Batch):
      bsize= mu_prime_Batch.size(0)
      mu_prime_Batch = mu_prime_Batch.to(self.device_cpu)


      reshaped_tensor = mu_prime_Batch.view(bsize * self.num_groups*self.num_groups_per_pair,
                               self.num_tokens, self.embedding_dim)

      # Step 2: Apply batch normalization
      batch_norm = nn.BatchNorm1d(num_features=self.num_tokens * self.embedding_dim, affine=False)
      batch_norm= batch_norm.to(self.device_cpu)
      normalized_tensor_flat = batch_norm(reshaped_tensor.view(-1, self.num_tokens * self.embedding_dim))

            # Step 4: Check for rows with all zeros along the self.num_tokens dimension
      for idx, row in enumerate(normalized_tensor_flat):
          if torch.all(reshaped_tensor[idx, :, 1] == 0):
              normalized_tensor_flat[idx, self.embedding_dim:2 * self.embedding_dim] = torch.zeros_like(row[self.embedding_dim:2 * self.embedding_dim])

      # Reshape back to the original shape
      final_tensor = normalized_tensor_flat.view(bsize, self.num_groups, self.num_groups_per_pair,
                                            self.num_tokens, self.embedding_dim)
      return final_tensor

    def problem(self, X, actor, PLM):
        # X is expected to be of shape [num_candidates, num_features]

        X_reshaped = X.view(-1, self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
        print (torch.isnan(X_reshaped).any(),"X_reshaped.nan)")
        # Initialize a tensor to store the rewards
        batch_rewards = torch.empty(X_reshaped.size(0), 1, device=self.device)

        # Define a maximum number of retries to prevent infinite loops

        max_retries= 1
        # for i in range(X_reshaped.size(0)):
        retry_count = 0
        while retry_count< max_retries:
            noise_tensor = self.gaussian_noise_subspace(X_reshaped).detach()

            mu_prime_B = self.calculate_noisy_embeddings(self.group_embeddings.detach(), noise_tensor, True).to(self.device)
            #print ("mu_prime_B.size)", mu_prime_B.size())
            #flattened_mu = self.normalize_batch_mu(mu_prime_B)
            #mu_prime_B = flattened_mu.to(self.device)
            #print("mu_prime_B", mu_prime_B.size(), mu_prime_B )
            reward = self.reward.calculate_reward(mu_prime_B.detach(), actor, PLM, self.lambda1).detach()

            if not torch.isnan(reward).any():
                batch_rewards = reward
                break  # Exit the while loop if the reward is not nan
            retry_count += 1  # Increment the retry counter

        if retry_count == max_retries:
            # Handle the case where a valid reward could not be calculated after max retries
            print(f"Warning: Unable to calculate a valid reward for candidate after {max_retries} retries. Setting to default value.")
            #batch_rewards = torch.tensor([10.0], device=self.device)  # Set a default value for the reward
              # Example batch size
            fill_value = 10.0  # Value to fill the tensor with

            batch_rewards = torch.full((self.batch_size, 1), fill_value, device=self.device)


        return (-1/batch_rewards).view(-1, 1)
        #return batch_rewards.view(-1, 1)



    def problem_objective(self, X, actor, PLM):
        gc.collect()
        torch.cuda.empty_cache()
        # Convert X from NumPy array to PyTorch tensor with the same dimensionality
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        X_reshaped = X_tensor.view(-1, self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
        print(torch.isnan(X_reshaped).any(), "X_reshaped.nan)")
        
        # Initialize a tensor to store the rewards
        batch_rewards = torch.empty(X_reshaped.size(0), 1, device=self.device)
        
        max_retries = 1  # Define a maximum number of retries to prevent infinite loops
        
        noise_tensor = self.gaussian_noise_subspace(X_reshaped).detach()
        
        mu_prime_B = self.calculate_noisy_embeddings(self.group_embeddings.detach(), noise_tensor, True).to(self.device)
        
        # Calculate rewards
        reward = self.reward.calculate_reward(mu_prime_B.detach(), actor, PLM, self.lambda1, True, True).detach()
        
        batch_rewards = reward
        
        # Convert the final PyTorch tensor back to a NumPy array with the same dimensionality
        final_result = (-1 / batch_rewards).view(-1, 2).cpu().numpy()
        
        return final_result



    # def run_bayesian_optimization(self,actor, PLM, candid):
    #     # Define arguments directly
            
    #     import logging

    #     # Configure logging to output to terminal with level DEBUG
    #     logging.basicConfig(level=logging.DEBUG,
    #                         format='%(asctime)s - %(levelname)s - %(message)s')

    #     # Example usage
    #     logging.debug('This is a debug message')
    #     logging.info('This is an info message')
    #     logging.warning('This is a warning message')
    #     logging.error('This is an error message')
    #     logging.critical('This is a critical message')
    #     args = Namespace(
    #         problem='myproblem',
    #         n_var=self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups,
    #         n_obj=2,
    #         n_init_sample=16,
    #         n_iter=10,
    #         ref_point=None,
    #         batch_size=16,
    #         seed=0,
    #         n_seed=1,
    #         algo='dgemo',
    #         subfolder='default',
    #         exp_name=None,
    #         log_to_file=True,
    #         n_process=1,  # or any other default value you prefer
    #         # Add any other arguments needed by your framework
    #     )
        
    #     # Define framework_args directly if needed
    #     framework_args = {
    #         'surrogate': {
    #             'surrogate': 'gp',
    #             'n_spectral_pts': 100,
    #             'nu': 5,
    #             'mean_sample': False,
    #             # Add any other surrogate model arguments
    #         },
    #         'acquisition': {
    #             'acquisition': 'identity',
    #             # Add any other acquisition function arguments
    #         },
    #         'solver': {
    #             'solver': 'discovery',
    #             'pop_size': 100,
    #             'n_gen': 10,
    #             'pop_init_method': 'lhs',
    #             'batch_size': 16,  # Make sure to not conflict with general args
    #             # Add any other solver arguments
    #         },
    #         'selection': {
    #             'selection': 'hvi',
    #             'batch_size': 16,
    #             # Add any other selection method arguments
    #         },
    #         # Include other components as needed
        
    #     }
    #     np.random.seed(args.seed)
    #     self.group_embeddings = self.calculate_embeddings(actor)
    #     gc.collect()

    #     #??logger and probkem need to be self?
    #     if  self.start_cond ==True:
    #     # build problem, get initial samples
    #         self.problem_f, true_pfront, X_init, Y_init = build_problem(self, args.problem, args.n_var, args.n_obj, args.n_init_sample,  actor, PLM, args.n_process)
    #         args.n_var, args.n_obj = self.problem_f.n_var, self.problem_f.n_obj

    #         # initialize optimizer
    #         self.optimizer = get_algorithm(args.algo)(self.problem_f, args.n_iter, args.ref_point, framework_args)

    #         # save arguments & setup logger
    #         #save_args(args, framework_args)
    #         self.logger = setup_logger(args)
    #         print(self.problem_f, self.optimizer, sep='\n')
            
    #         # initialize data exporter
    #         self.exporter = DataExport(self.optimizer, X_init, Y_init, args)

    #         # optimization
    #         self.solution = self.optimizer.solve(X_init, Y_init)

    #     # # export true Pareto front to csv
    #     # if true_pfront is not None:
    #     #     exporter.write_truefront_csv(true_pfront)

    #         if self.start_cond== True:
    #             n_iter= 1
    #             self.start_cond = False
    #         else:
    #             n_iter= 1       


    #     y = []
    #     some_candidates = np.array([]).reshape(0, self.problem_f.n_var)  # Assuming `problem.n_var` is defined

    #     for _ in range(n_iter):
    #         # get new design samples and corresponding performance
    #         X_next, Y_next = next(self.solution)
    #         gc.collect()
    #         torch.cuda.empty_cache()
            
    #         # update & export current status to csv
    #         self.exporter.update(X_next, Y_next)
    #         self.exporter.write_csvs()
            


    #         batchNoiseEval = Y_next[:, 0]
    #         perf_eval = Y_next[:, 1]
            
            
    # # Convert Y_next to a PyTorch tensor if it's not already
    #         Y_next_tensor = torch.tensor(Y_next, dtype=torch.float32).to(self.device)

    #         # Extract batchNoiseEval and perf_eval as tensors
    #         batchNoiseEval = Y_next_tensor[:, 0]
    #         perf_eval = Y_next_tensor[:, 1]

    #         # Calculate losses and adjust using CoVWeighting, assuming adjust_loss_weights can handle tensor input
    #         losses_tensor = torch.tensor([torch.mean(batchNoiseEval), torch.mean(perf_eval)], dtype=torch.float32)
    #         _, weights = self.CoVWeighting.adjust_loss_weights(losses_tensor)

    #         # Compute adjusted_Y using the weights
    #         adjusted_Y = batchNoiseEval * weights[0] - perf_eval * weights[1]

    #         # Initialize y and some_candidates as tensors if they are not already defined
    #         if 'y' not in locals():
    #             y = torch.empty((0,), dtype=torch.float32).to(self.device)
    #         if 'some_candidates' not in locals():
    #             some_candidates = torch.empty((0, X_next.shape[1]), dtype=torch.float32).to(self.device)

    #         # Convert X_next to a PyTorch tensor if it's not already
    #         X_next_tensor = torch.tensor(X_next, dtype=torch.float32).to(self.device)

    #         # Append current batch's adjusted performance (y) and candidates (some_candidates)
    #         y = torch.cat((y, adjusted_Y.unsqueeze(0)), dim=0)
    #         some_candidates = torch.cat((some_candidates, X_next_tensor), dim=0)

            
    #     # Convert y, some_candidates to torch tensors with same dimensionality

    #     # Selection of the best candidates based on adjusted performance
    #     sorted_indices = torch.argsort(y, descending=True)  # Use ascending=False to minimize
    #     best_indices = sorted_indices[:self.batch_size]

    #     self.best_X = some_candidates[best_indices]
    #     self.best_Y = y[best_indices]

    #     # Detach and reshape for consistency (assuming they are already in the correct device and tensor format)
    #     self.best_Y = self.best_Y.view(-1).squeeze(-1)

    #     # Assuming `unnormalize`, `calculate_mu_prime_B`, `calculate_noisy_embeddings`, and `normalize_batch_mu`
    #     # are defined elsewhere along with `self.group_embeddings`, `actor`, `PLM`, `self.device`

    #     # Final processing and return statement
    #     lower_bound = -1.0
    #     upper_bound = 1.0

    #     # Create a tensor with the bounds for each dimension
    #     #bounds_tensor = torch.tensor([[lower_bound, upper_bound]] * self.embedding_dim, device=self.device)
    #     bounds_tensor = torch.tensor([lower_bound, upper_bound]).repeat(self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups, 1).double()

    #     self.best_X = unnormalize(self.best_X.cpu().numpy(), bounds_tensor)  # Assuming `bounds` is defined
    #     batch_mu = self.calculate_mu_prime_B(self.best_X, actor, PLM)
    #     self.mu_prime_Batch = self.calculate_noisy_embeddings(self.group_embeddings, batch_mu, True).to(self.device)
    #     flattened_mu = self.normalize_batch_mu(self.mu_prime_Batch)
        
    #     self.mu_prime_Batch = flattened_mu.to(self.device)

    #     # Close logger if applicable
    #     if self.logger is not None:
    #         self.logger.close()


    #     gc.collect()
    #     return self.best_X






    # this is the orginal run bayesian optimization fnction
    def run_bayesian_optimization(self, actor, PLM, candid, num_samples=100, maxiter=5, improvement_threshold=0.0001, mixed_sampling_weight=0.2, num_restarts=2, raw_samples=100, batch_limit=8):

                best_value = float('-inf')
                self.samples = []
                total_num_tasks = 1#self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups

                print ("entering the function bo")
                print_gpu_usage()
                upsampleq=3
                gc.collect()
                torch.cuda.empty_cache()
                device_cpu = torch.device("cpu")
                N_W = raw_samples
                STD_DEV = 0.05
                ALPHA = 0.8
                #risk_measure = VaR(alpha=ALPHA, n_w=self.batch_size)
                tkwargs = {"device": "cpu", "dtype": torch.double}
                #likelihood = GaussianLikelihood()
                #while True:
                lower_bound = -1.0
                upper_bound = 1.0

                # Create a tensor with the bounds for each dimension
                #bounds_tensor = torch.tensor([[lower_bound, upper_bound]] * self.embedding_dim, device=self.device)
                bounds_tensor = torch.tensor([lower_bound, upper_bound]).repeat(self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups, 1).double()

                # Transpose the tensor to get the desired shape
                bounds_tensor = bounds_tensor.T
                bounds_tensor=bounds_tensor.to(device_cpu)
                self.group_embeddings = self.calculate_embeddings(actor)
                if self.start_cond == True:
                    self.X_all = torch.empty(0, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()
                    self.Y_all = torch.empty(0,1).double()
                    self.best_X= torch.empty(0, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()
                    self.best_Y= torch.empty(0,1).double()
                    self.X_all=self.X_all.to(device_cpu)

                    #init_x = self.default_values  # Ensure init_x is double
                    init_x = self.initialize_data()
                    init_x=init_x.to(device_cpu)
                    print ("init_x", init_x.size())

                    #init_x_normalized = self.normalize(init_x, self.bounds)
                    with torch.no_grad():

                      init_y = self.problem(init_x, actor, PLM).detach().double()

                    init_y = init_y.view(init_x.size(0), 1)  # Ensure init_y is double
                    print(init_x.size(), init_y.size(), "final")
                    #print (init_x,"init_x" ,init_y,"init_y" )
                    #init_y= standardize(init_y)
                    #print ("dev",init_y.device)
                    init_y=init_y.to(device_cpu )
                    #init_x=self.normalize(init_x, self.bounds)
                    init_x_flat = init_x.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).detach().double()  # Ensure init_x_flat is double
                    task_feature=init_x_flat.shape[-1] - 1
                    #likelihood = MultitaskGaussianLikelihood(num_tasks=total_num_tasks)


                    inpf=Normalize(d=init_x_flat.size(1))
                    # intf = InputPerturbation(
                    # perturbation_set=draw_sobol_normal_samples(d=init_x_flat.size(1), n=self.batch_size, **tkwargs) * STD_DEV,
                    # bounds=bounds_tensor,
                    # )
                    if self.model== None:
                        self.model = SingleTaskGP(init_x_flat, init_y,input_transform=inpf, outcome_transform=Standardize(m=1))
                        #self.model = SingleTaskGP(init_x_flat, init_y,input_transform=inpf, outcome_transform=Standardize(m=1))
                        self.model.double()  # Ensure the model parameters are double
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                        fit_gpytorch_model(mll)

                    self.X_all = torch.cat([self.X_all, init_x_flat], dim=0)
                    self.Y_all= self.Y_all.to(device_cpu)
                    #print ("alldev",self.device,self.Y_all.device, init_y.device )
                    #self.Y_all = torch.cat([self.Y_all, init_y.detach().squeeze(-1)], dim=0)#.view(-1, 1)
                    self.Y_all = torch.cat([self.Y_all, init_y.detach()], dim=0)#.view(-1, 1)
                    
                    print ("sizes", init_x_flat.size(), init_y.size(), self.X_all.size(), self.Y_all.size())
                    #self.model.set_

                    self.best_value = best_value
                    self.start_cond = False


                else:
                    init_x = candid.double()  # Ensure candid is double
                    candid = candid.to('cpu')

                    # Ensure init_y is double
                    #init_x_normalized = self.normalize(init_x, self.bounds)

                    #init_x_flat = init_x_normalized.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()  # Ensure init_x_flat is double
                    init_x_flat = init_x.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()  # Ensure init_x_flat is double
                    # print (init_x_flat.size(),"init_x_flat.size()")
                    # task_feature=init_x_flat.shape[-1] - 1
                    # #
                    # init_y = self.problem(init_x_flat, actor, PLM).double()
                    # init_y= standardize(init_y + 0.05 * torch.randn_like(init_y))
                    # #init_y = standardize(init_y + 1e-1 * torch.randn_like(init_y))



                d = init_x_flat.shape[-1]

                self.model.to(device_cpu )
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

                #self.model.likelihood.to(device_cpu)
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
                #risk_measure = VaR(alpha=ALPHA, n_w=min(len(self.X_all), num_samples))
                acqf = qNoisyExpectedImprovement(
                    model=self.model,
                    X_baseline=self.X_all,
                    sampler=sampler,
                 #   objective=risk_measure,
                    prune_baseline=True,
                )
                # if len (self.X_all)>num_samples:

                #     acqf = qNoisyExpectedImprovement(
                #     model=self.model,
                #     X_baseline=self.X_all,
                #     sampler=sampler,
                #     objective=risk_measure,
                #     prune_baseline=True,
                # )
                print ("finished acq")
                candidate, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds_tensor,
                    q=self.batch_size*upsampleq,
                    num_restarts=num_restarts,
                    raw_samples=min (self.X_all.size(0), raw_samples),
                )
                #print ("candidate",candidate)
                #new_observations = evaluate_function(candidate)
                candidates = candidate.detach()  # Detach once here
                torch.cuda.empty_cache()
                new_y_values = []

                #candidate_reshaped = candidates_flat.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
                candidate_reshaped = candidates.view(self.batch_size * upsampleq,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
                #self.candidates =self.unnormalize(candidate_reshaped, self.bounds).detach().clone()
                self.candidates =unnormalize(candidate, bounds_tensor).view(self.batch_size * upsampleq,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
                self.candidates.requires_grad_(False)
                candidate_unnormalized= unnormalize(candidate, bounds_tensor)
                new_y_list = []
                new_mu_list = []
                with torch.no_grad():

                  for i in range(upsampleq):
                      # Selecting a batch of candidates
                      batch_candidates = self.candidates[i * self.batch_size:(i + 1) * self.batch_size]

                      # Process the batch and get results
                      batch_y = self.problem(batch_candidates.detach(), actor, PLM).detach()
                      batch_mu = self.calculate_mu_prime_B(  batch_candidates.detach(), actor, PLM).detach()
                      # Append results to the list
                      new_y_list.append(batch_y)
                      #flattened_mu = self.normalize_batch_mu(self.mu_prime_Batch)
                      flattened_mu = self.normalize_batch_mu(batch_mu)
                      #self.mu_prime_Batch = flattened_mu.to(self.device)

                      #print ("flattened_mu.size()",flattened_mu.size())
                      new_mu_list.append(flattened_mu)

                # Concatenate all the results along the first dimension
                new_y_flat = torch.cat(new_y_list, dim=0)
                new_mu_flat = torch.cat(new_mu_list, dim=0)
                #print ("new_mu_flat.size()", new_mu_flat.size())
                #new_y_flat = self.problem(self.candidates.detach(), actor, PLM

                #new_y_flat = self.problem(self.candidates.detach(), actor, PLM).detach()
                            # Assuming new_y_flat contains the objective values for each candidate and you want to maximize these values
                sorted_indices = torch.argsort(new_y_flat, descending=True)  # Change to ascending=False if you want to minimize

                # Select the top batch_size candidates
                candidates=candidates.to(device_cpu)
                best_indices = sorted_indices[:self.batch_size]
                best_indices=best_indices.to(device_cpu)
                best_mu = new_mu_flat[best_indices].squeeze(dim=1)

                best_mu = best_mu.to(device_cpu)
                #best_mu = self.mu_prime_Batch[best_indices]

                #print ("candidates",candidates)
                self.best_X = candidates[best_indices]
                print ("self.best_X.shape",self.best_X.shape)

                new_y_flat = new_y_flat.to(device_cpu).view(upsampleq*self.batch_size,1)



                #new_y_flat_standardized=standardize(new_y_flat)#+ 0.05 * torch.randn_like(new_y_flat))
                self.best_Y = new_y_flat[best_indices]

                # Detach and reshape for consistency
                self.best_X = self.best_X #.detach().clone()
                self.best_Y = self.best_Y.detach().clone().view(-1).squeeze(-1)

                # Update X_all and Y_all with all candidates and their corresponding y values
                self.X_all=self.X_all.clone()
                self.X_all = torch.cat([self.X_all, candidate_unnormalized.detach()], dim=0)
                #print ("this", self.Y_all.shape , new_y_flat_standardized.detach().view(-1).squeeze(-1), new_y_flat_standardized.detach().view(-1).squeeze(-1).shape )
                #self.Y_all = torch.cat([self.Y_all, new_y_flat_standardized.detach().view(-1).squeeze(-1)], dim=0)
                #self.Y_all==self.Y_all.clone()
                #self.Y_all = torch.cat([self.Y_all.clone(), new_y_flat.detach().view(-1).squeeze(-1)], dim=0)
                #self.Y_all = torch.cat([self.Y_all.clone(), new_y_flat.detach().squeeze(-1)], dim=0)
                self.Y_all = torch.cat([self.Y_all.clone(), new_y_flat.detach()], dim=0)
                self.Y_all = self.Y_all#.view(-1,1)
                #self.Y_all = torch.cat([self.Y_all, new_y_flat_standardized.detach()], dim=0)
                #print ("Y_all.shape, new_y_flat.shape",self.Y_all.shape, new_y_flat.shape, new_y_flat.view(-1).squeeze(-1).shape)

                self.X_all = self.X_all#.detach()
                self.Y_all = self.Y_all#.detach()
                self.X_all = self.X_all.double()
                self.Y_all = self.Y_all.double()

                # Set the new training data for the model
                #self.Y_all_standardized = standardize(self.Y_all + 0.05 * torch.randn_like(self.Y_all))
                #self.Y_all.requires_grad = True
                #likelihood = GaussianLikelihood()
                #print ("that",init_x_flat.shape, init_y.shape)
                #self.model = SingleTaskGP(init_x_flat, init_y, likelihood= likelihood, input_transform=intf, outcome_transform=Standardize(m=1))
                self.model.double()  # Ensure the model parameters are double
                # Reshape Y from [n] to [n, 1]
                # intf = InputPerturbation(
                #     perturbation_set=draw_sobol_normal_samples(d=self.X_all.shape[-1], n=min ( num_samples,self.Y_all.size(0)), **tkwargs) * STD_DEV,
                #     bounds=bounds_tensor,
                #     )

                # #self.model = SingleTaskGP(self.X_all, self.Y_all.clone().view(-1, 1), input_transform=intf, outcome_transform=Standardize(m=1))
                #self.model = SingleTaskGP(self.X_all, self.Y_all, input_transform=intf, outcome_transform=Standardize(m=1))
                inpf=Normalize(d=self.X_all.shape[-1])
                self.model.to(device_cpu)
                self.X_all = self.X_all.to(device_cpu)
                self.Y_all = self.Y_all.to(device_cpu)
                self.best_X = self.best_X.to(device_cpu)
                self.best_Y = self.best_Y.to(device_cpu)

                n_inducing_points = 10  # Number of inducing points you want to select

                # Select inducing points based on the highest Y_all values
                _, top_indices = torch.topk(self.Y_all.squeeze(), min(1500, self.X_all.size(0)))
                inducing_Xpoints = self.X_all[top_indices]
                inducing_Ypoints = self.Y_all[top_indices]
                self.Y_all= inducing_Ypoints
                self.X_all= inducing_Xpoints
                
                
                # # Initialize model and likelihood
                # likelihood = GaussianLikelihood()
                # model = SparseGPModel(X_all, Y_all, likelihood, inducing_points)

                # # Fit the model
                # mll = ExactMarginalLogLikelihood(model.likelihood, model)
                # fit_gpytorch_model(mll)


                self.model = SingleTaskGP(self.X_all, self.Y_all, input_transform=inpf, outcome_transform=Standardize(m=1))
                self.model.set_train_data(inputs=self.X_all, targets=self.Y_all.clone().squeeze(-1), strict=False)


                # Optimize the model
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                fit_gpytorch_model(mll)

                # Check for significant improvement with the best values
                # Calculate the average of best_Y
 #               average_best_Y = torch.mean(self.best_Y)

                # Check if the absolute difference between the average of best_Y and the best_value is less than the threshold
                # if abs(average_best_Y - self.best_value) < improvement_threshold and self.best_value!=0:
                #     self.start_cond = True
                #     continue
                # else:
                #     self.best_value = torch.max(self.best_Y)  # Update self.best_value to the new average
                #     break


                # Process the entire batch of best candidates

                #best_reshaped = candidates[best_indices].view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups).detach()
                #self.best_X =self.unnormalize(best_reshaped, self.bounds).detach().clone()
                self.best_X =unnormalize(self.best_X, bounds_tensor).detach().clone()#.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
                #best_X_reshaped= self.best_X.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
                #noise_tensor = self.gaussian_noise_subspace(best_X_reshaped)
                #self.mu_prime_Batch = self.calculate_noisy_embeddings(self.group_embeddings, noise_tensor, True).to(self.device)
                #print("best_mu.size()",best_mu.size())
                self.mu_prime_Batch = self.calculate_noisy_embeddings(self.group_embeddings.detach(), best_mu, True).to(self.device)
                del candidates, batch_candidates, batch_y, new_y_flat, new_y_list
                torch.cuda.empty_cache()
                gc.collect()
                return self.best_X




        #this one s functional but too slow.
    # def run_bayesian_optimization(self, actor, PLM, candid, num_samples=50, maxiter=2, improvement_threshold=0.0001, mixed_sampling_weight=0.2, num_restarts=2, raw_samples=50, batch_limit=8):

    #         best_value = float('-inf')
    #         self.samples = []
    #         total_num_tasks = 1#self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups

    #         print ("entering the function bo")
    #         print_gpu_usage()
    #         upsampleq=3
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         device_cpu = torch.device("cpu")
    #         N_W = raw_samples
    #         NUM_ITERATIONS = 25
    #         STD_DEV = 0.05
    #         ALPHA = 0.8
    #         risk_measure = VaR(alpha=ALPHA, n_w=self.batch_size)
    #         tkwargs = {"device": "cpu", "dtype": torch.double}
    #         #likelihood = GaussianLikelihood()
    #         #while True:
    #         bounds_tensor = torch.tensor([torch.min(self.bounds), torch.max(self.bounds)]).repeat(self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups, 1).double()

    #         # Transpose the tensor to get the desired shape
    #         bounds_tensor = bounds_tensor.T
    #         bounds_tensor=bounds_tensor.to(device_cpu)
    #         self.group_embeddings = self.calculate_embeddings(actor)
    #         if self.start_cond == True:
    #             self.X_all = torch.empty(0, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()
    #             self.Y_all = torch.empty(0,1).double()

    #             self.X_all=self.X_all.to(device_cpu)

    #             init_x = self.initialize_data().double()  # Ensure init_x is double
    #             init_x=init_x.to(device_cpu)

    #             #init_x_normalized = self.normalize(init_x, self.bounds)

    #             init_y = self.problem(init_x, actor, PLM).detach().double()
    #             init_y = init_y.view(init_x.size(0), 1)  # Ensure init_y is double
    #             print(init_x.size(), init_y.size(), "final")
    #             #print (init_x,"init_x" ,init_y,"init_y" )
    #             #init_y= standardize(init_y)
    #             #print ("dev",init_y.device)
    #             init_y=init_y.to(device_cpu )
    #             #init_x=self.normalize(init_x, self.bounds)
    #             init_x_flat = init_x.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).detach().double()  # Ensure init_x_flat is double
    #             task_feature=init_x_flat.shape[-1] - 1
    #             #likelihood = MultitaskGaussianLikelihood(num_tasks=total_num_tasks)


    #             #inpf=input_transform=Normalize(d=init_x_flat.size(1))
    #             intf = InputPerturbation(
    #             perturbation_set=draw_sobol_normal_samples(d=init_x_flat.size(1), n=self.batch_size, **tkwargs) * STD_DEV,
    #             bounds=bounds_tensor,
    #             )

    #             self.model = SingleTaskGP(init_x_flat, init_y,input_transform=intf, outcome_transform=Standardize(m=1))
    #             #self.model = SingleTaskGP(init_x_flat, init_y,input_transform=inpf, outcome_transform=Standardize(m=1))
    #             self.model.double()  # Ensure the model parameters are double

    #             self.X_all = torch.cat([self.X_all, init_x_flat], dim=0)
    #             self.Y_all= self.Y_all.to(device_cpu)
    #             #print ("alldev",self.device,self.Y_all.device, init_y.device )
    #             #self.Y_all = torch.cat([self.Y_all, init_y.detach().squeeze(-1)], dim=0)#.view(-1, 1)
    #             self.Y_all = torch.cat([self.Y_all, init_y.detach()], dim=0)#.view(-1, 1)

    #             print ("sizes", init_x_flat.size(), init_y.size(), self.X_all.size(), self.Y_all.size())
    #             #self.model.set_
    #             mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    #             fit_gpytorch_model(mll)
    #             self.best_value = best_value
    #             self.start_cond = False


    #         else:
    #             init_x = candid.double()  # Ensure candid is double
    #               # Ensure init_y is double
    #             #init_x_normalized = self.normalize(init_x, self.bounds)

    #             #init_x_flat = init_x_normalized.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()  # Ensure init_x_flat is double
    #             init_x_flat = init_x.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()  # Ensure init_x_flat is double
    #             # print (init_x_flat.size(),"init_x_flat.size()")
    #             # task_feature=init_x_flat.shape[-1] - 1
    #             # #
    #             # init_y = self.problem(init_x_flat, actor, PLM).double()
    #             # init_y= standardize(init_y + 0.05 * torch.randn_like(init_y))
    #             # #init_y = standardize(init_y + 1e-1 * torch.randn_like(init_y))



    #         d = init_x_flat.shape[-1]

    #         self.model.to(device_cpu )
    #         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    #         #self.model.likelihood.to(device_cpu)
    #         sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
    #         risk_measure = VaR(alpha=ALPHA, n_w=min(len(self.X_all), num_samples))
    #         acqf = qNoisyExpectedImprovement(
    #             model=self.model,
    #             X_baseline=self.X_all,
    #             sampler=sampler,
    #             objective=risk_measure,
    #             prune_baseline=True,
    #         )
    #         # if len (self.X_all)>num_samples:

    #         #     acqf = qNoisyExpectedImprovement(
    #         #     model=self.model,
    #         #     X_baseline=self.X_all,
    #         #     sampler=sampler,
    #         #     objective=risk_measure,
    #         #     prune_baseline=True,
    #         # )
    #         print ("finished acq")
    #         candidate, _ = optimize_acqf(
    #             acq_function=acqf,
    #             bounds=bounds_tensor,
    #             q=self.batch_size*upsampleq,
    #             num_restarts=num_restarts,
    #             raw_samples=min (self.X_all.size(0), raw_samples),
    #         )

    #         #new_observations = evaluate_function(candidate)
    #         candidates = candidate.detach()  # Detach once here
    #         torch.cuda.empty_cache()
    #         new_y_values = []

    #         #candidate_reshaped = candidates_flat.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #         candidate_reshaped = candidates.view(self.batch_size * upsampleq,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #         #self.candidates =self.unnormalize(candidate_reshaped, self.bounds).detach().clone()
    #         self.candidates =unnormalize(candidate, bounds_tensor).view(self.batch_size * upsampleq,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #         self.candidates.requires_grad_(False)
    #         candidate_unnormalized= unnormalize(candidate, bounds_tensor)
    #         new_y_list = []

    #         for i in range(upsampleq):
    #             # Selecting a batch of candidates
    #             batch_candidates = self.candidates[i * self.batch_size:(i + 1) * self.batch_size]

    #             # Process the batch and get results
    #             batch_y = self.problem(batch_candidates.detach(), actor, PLM).detach()

    #             # Append results to the list
    #             new_y_list.append(batch_y)

    #         # Concatenate all the results along the first dimension
    #         new_y_flat = torch.cat(new_y_list, dim=0)

    #         #new_y_flat = self.problem(self.candidates.detach(), actor, PLM).detach()
    #                     # Assuming new_y_flat contains the objective values for each candidate and you want to maximize these values
    #         sorted_indices = torch.argsort(new_y_flat, descending=True)  # Change to ascending=False if you want to minimize

    #         # Select the top batch_size candidates
    #         candidates=candidates.to(device_cpu)
    #         best_indices = sorted_indices[:self.batch_size]
    #         best_indices=best_indices.to(device_cpu)

    #         #print ("candidates",candidates)
    #         self.best_X = candidates[best_indices]
    #         print ("self.best_X.shape",self.best_X.shape)

    #         new_y_flat = new_y_flat.to(device_cpu).view(upsampleq*self.batch_size,1)



    #         #new_y_flat_standardized=standardize(new_y_flat)#+ 0.05 * torch.randn_like(new_y_flat))
    #         self.best_Y = new_y_flat[best_indices]

    #         # Detach and reshape for consistency
    #         self.best_X = self.best_X #.detach().clone()
    #         self.best_Y = self.best_Y.detach().clone().view(-1).squeeze(-1)

    #         # Update X_all and Y_all with all candidates and their corresponding y values
    #         self.X_all=self.X_all.clone()
    #         self.X_all = torch.cat([self.X_all, candidate_unnormalized.detach()], dim=0)
    #         #print ("this", self.Y_all.shape , new_y_flat_standardized.detach().view(-1).squeeze(-1), new_y_flat_standardized.detach().view(-1).squeeze(-1).shape )
    #         #self.Y_all = torch.cat([self.Y_all, new_y_flat_standardized.detach().view(-1).squeeze(-1)], dim=0)
    #         #self.Y_all==self.Y_all.clone()
    #         #self.Y_all = torch.cat([self.Y_all.clone(), new_y_flat.detach().view(-1).squeeze(-1)], dim=0)
    #         #self.Y_all = torch.cat([self.Y_all.clone(), new_y_flat.detach().squeeze(-1)], dim=0)
    #         self.Y_all = torch.cat([self.Y_all.clone(), new_y_flat.detach()], dim=0)
    #         self.Y_all = self.Y_all#.view(-1,1)
    #         #self.Y_all = torch.cat([self.Y_all, new_y_flat_standardized.detach()], dim=0)
    #         #print ("Y_all.shape, new_y_flat.shape",self.Y_all.shape, new_y_flat.shape, new_y_flat.view(-1).squeeze(-1).shape)

    #         self.X_all = self.X_all#.detach()
    #         self.Y_all = self.Y_all#.detach()
    #         self.X_all = self.X_all.double()
    #         self.Y_all = self.Y_all.double()

    #         # Set the new training data for the model
    #         #self.Y_all_standardized = standardize(self.Y_all + 0.05 * torch.randn_like(self.Y_all))
    #         #self.Y_all.requires_grad = True
    #         #likelihood = GaussianLikelihood()
    #         #print ("that",init_x_flat.shape, init_y.shape)
    #         #self.model = SingleTaskGP(init_x_flat, init_y, likelihood= likelihood, input_transform=intf, outcome_transform=Standardize(m=1))
    #         self.model.double()  # Ensure the model parameters are double
    #         # Reshape Y from [n] to [n, 1]
    #         intf = InputPerturbation(
    #             perturbation_set=draw_sobol_normal_samples(d=self.X_all.shape[-1], n=min ( num_samples,self.Y_all.size(0)), **tkwargs) * STD_DEV,
    #             bounds=bounds_tensor,
    #             )

    #         # #self.model = SingleTaskGP(self.X_all, self.Y_all.clone().view(-1, 1), input_transform=intf, outcome_transform=Standardize(m=1))
    #         self.model = SingleTaskGP(self.X_all, self.Y_all, input_transform=intf, outcome_transform=Standardize(m=1))
    #         #inpf=Normalize(d=self.X_all.shape[-1])
    #         self.model = SingleTaskGP(self.X_all, self.Y_all, input_transform=intf, outcome_transform=Standardize(m=1))
    #         self.model.set_train_data(inputs=self.X_all, targets=self.Y_all.clone().squeeze(-1), strict=False)

    #         # Optimize the model
    #         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
    #         fit_gpytorch_model(mll)

    #         # Check for significant improvement with the best values
    #         # Calculate the average of best_Y
    #         average_best_Y = torch.mean(self.best_Y)

    #         # Check if the absolute difference between the average of best_Y and the best_value is less than the threshold
    #         # if abs(average_best_Y - self.best_value) < improvement_threshold and self.best_value!=0:
    #         #     self.start_cond = True
    #         #     continue
    #         # else:
    #         #     self.best_value = torch.max(self.best_Y)  # Update self.best_value to the new average
    #         #     break


    #         # Process the entire batch of best candidates

    #         #best_reshaped = candidates[best_indices].view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups).detach()
    #         #self.best_X =self.unnormalize(best_reshaped, self.bounds).detach().clone()
    #         self.best_X =unnormalize(self.best_X, bounds_tensor).detach().clone()#.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #         best_X_reshaped= self.best_X.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #         noise_tensor = self.gaussian_noise_subspace(best_X_reshaped)
    #         self.mu_prime_Batch = self.calculate_noisy_embeddings(self.group_embeddings, noise_tensor, True).to(self.device)

    #         del candidates, batch_candidates, batch_y, noise_tensor, new_y_flat, new_y_list
    #         torch.cuda.empty_cache()
    #         # self.best_X = self.best_X.to(self.device)
    #         # self.best_Y = self.best_Y.to(self.device)
    #         return self.best_X



    # def run_bayesian_optimization(self, actor, PLM, candid, num_samples=50, maxiter=2, improvement_threshold=0.0001, mixed_sampling_weight=0.2, num_restarts=2, raw_samples=50, batch_limit=8):

    #     best_value = float('-inf')
    #     self.samples = []
    #     total_num_tasks = 1#self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups

    #     print ("entering the function bo")
    #     print_gpu_usage()
    #     upsampleq=3
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     device_cpu = torch.device("cpu")
    #     N_W = raw_samples
    #     NUM_ITERATIONS = 25
    #     STD_DEV = 0.05
    #     ALPHA = 0.8
    #     risk_measure = VaR(alpha=ALPHA, n_w=self.batch_size)
    #     tkwargs = {"device": "cpu", "dtype": torch.double}
    #     likelihood = GaussianLikelihood()
    #     #while True:
    #     bounds_tensor = torch.tensor([torch.min(self.bounds), torch.max(self.bounds)]).repeat(self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups, 1).double()

    #     # Transpose the tensor to get the desired shape
    #     bounds_tensor = bounds_tensor.T
    #     bounds_tensor=bounds_tensor.to(device_cpu)
    #     self.group_embeddings = self.calculate_embeddings(actor)
    #     if self.start_cond == True:
    #         self.X_all = torch.empty(0, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()
    #         self.Y_all = torch.empty(0).double()

    #         self.X_all=self.X_all.to(device_cpu)

    #         init_x = self.initialize_data().double()  # Ensure init_x is double
    #         init_x=init_x.to(device_cpu)

    #         #init_x_normalized = self.normalize(init_x, self.bounds)

    #         init_y = self.problem(init_x, actor, PLM).detach().double()  # Ensure init_y is double
    #         print(init_x.size(), init_y.size(), "final")
    #         #print (init_x,"init_x" ,init_y,"init_y" )
    #         #init_y= standardize(init_y)
    #         #print ("dev",init_y.device)
    #         init_y=init_y.to(device_cpu )
    #         #init_x=self.normalize(init_x, self.bounds)
    #         init_x_flat = init_x.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).detach().double()  # Ensure init_x_flat is double
    #         task_feature=init_x_flat.shape[-1] - 1
    #         #likelihood = MultitaskGaussianLikelihood(num_tasks=total_num_tasks)



    #         intf = InputPerturbation(
    #         perturbation_set=draw_sobol_normal_samples(d=init_x_flat.size(1), n=self.batch_size, **tkwargs) * STD_DEV,
    #         bounds=bounds_tensor,
    #         )

    #         self.model = SingleTaskGP(init_x_flat, init_y,likelihood=likelihood,input_transform=intf, outcome_transform=Standardize(m=1))
    #         self.model.double()  # Ensure the model parameters are double

    #         self.X_all = torch.cat([self.X_all, init_x_flat], dim=0)
    #         self.Y_all= self.Y_all.to(device_cpu)
    #         #print ("alldev",self.device,self.Y_all.device, init_y.device )
    #         self.Y_all = torch.cat([self.Y_all, init_y.detach().squeeze(-1)], dim=0)#.view(-1, 1)

    #         print ("sizes", init_x_flat.size(), init_y.size(), self.X_all.size(), self.Y_all.size())
    #         #self.model.set_
    #         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    #         fit_gpytorch_model(mll)
    #         self.best_value = best_value
    #         self.start_cond = False


    #     else:
    #         init_x = candid.double()  # Ensure candid is double
    #           # Ensure init_y is double
    #         #init_x_normalized = self.normalize(init_x, self.bounds)

    #         #init_x_flat = init_x_normalized.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()  # Ensure init_x_flat is double
    #         init_x_flat = init_x.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()  # Ensure init_x_flat is double
    #         # print (init_x_flat.size(),"init_x_flat.size()")
    #         # task_feature=init_x_flat.shape[-1] - 1
    #         # #
    #         # init_y = self.problem(init_x_flat, actor, PLM).double()
    #         # init_y= standardize(init_y + 0.05 * torch.randn_like(init_y))
    #         # #init_y = standardize(init_y + 1e-1 * torch.randn_like(init_y))



    #     d = init_x_flat.shape[-1]

    #     self.model.to(device_cpu )
    #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    #     #self.model.likelihood.to(device_cpu)
    #     sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
    #     risk_measure = VaR(alpha=ALPHA, n_w=min(len (self.X_all), num_samples))
    #     acqf = qNoisyExpectedImprovement(
    #         model=self.model,
    #         X_baseline=self.X_all,
    #         sampler=sampler,
    #         objective=risk_measure,
    #         prune_baseline=True,
    #     )
    #     if len (self.X_all)>num_samples:

    #         acqf = qNoisyExpectedImprovement(
    #         model=self.model,
    #         X_baseline=self.X_all,
    #         sampler=sampler,
    #         objective=risk_measure,
    #         prune_baseline=True,
    #     )

    #     candidate, _ = optimize_acqf(
    #         acq_function=acqf,
    #         bounds=bounds_tensor,
    #         q=self.batch_size*upsampleq,
    #         num_restarts=num_restarts,
    #         raw_samples=raw_samples,
    #     )

    #     #new_observations = evaluate_function(candidate)
    #     candidates = candidate.detach()  # Detach once here
    #     torch.cuda.empty_cache()
    #     new_y_values = []

    #     #candidate_reshaped = candidates_flat.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #     candidate_reshaped = candidates.view(self.batch_size * upsampleq,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #     #self.candidates =self.unnormalize(candidate_reshaped, self.bounds).detach().clone()
    #     self.candidates =unnormalize(candidate, bounds_tensor).view(self.batch_size * upsampleq,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #     self.candidates.requires_grad_(False)
    #     candidate_unnormalized= unnormalize(candidate, bounds_tensor)
    #     new_y_list = []

    #     for i in range(upsampleq):
    #         # Selecting a batch of candidates
    #         batch_candidates = self.candidates[i * self.batch_size:(i + 1) * self.batch_size]

    #         # Process the batch and get results
    #         batch_y = self.problem(batch_candidates.detach(), actor, PLM).detach()

    #         # Append results to the list
    #         new_y_list.append(batch_y)

    #     # Concatenate all the results along the first dimension
    #     new_y_flat = torch.cat(new_y_list, dim=0)

    #     #new_y_flat = self.problem(self.candidates.detach(), actor, PLM).detach()
    #                 # Assuming new_y_flat contains the objective values for each candidate and you want to maximize these values
    #     sorted_indices = torch.argsort(new_y_flat, descending=True)  # Change to ascending=False if you want to minimize

    #     # Select the top batch_size candidates
    #     candidates=candidates.to(device_cpu)
    #     best_indices = sorted_indices[:self.batch_size]
    #     best_indices=best_indices.to(device_cpu)

    #     #print ("candidates",candidates)
    #     self.best_X = candidates[best_indices]
    #     print ("self.best_X.shape",self.best_X.shape)

    #     new_y_flat = new_y_flat.to(device_cpu)

    #     #new_y_flat_standardized=standardize(new_y_flat)#+ 0.05 * torch.randn_like(new_y_flat))
    #     self.best_Y = new_y_flat[best_indices]

    #     # Detach and reshape for consistency
    #     self.best_X = self.best_X #.detach().clone()
    #     self.best_Y = self.best_Y.detach().clone().view(-1).squeeze(-1)

    #     # Update X_all and Y_all with all candidates and their corresponding y values
    #     self.X_all=self.X_all.clone()
    #     self.X_all = torch.cat([self.X_all, candidate_unnormalized.detach()], dim=0)
    #     #print ("this", self.Y_all.shape , new_y_flat_standardized.detach().view(-1).squeeze(-1), new_y_flat_standardized.detach().view(-1).squeeze(-1).shape )
    #     #self.Y_all = torch.cat([self.Y_all, new_y_flat_standardized.detach().view(-1).squeeze(-1)], dim=0)
    #     #self.Y_all==self.Y_all.clone()
    #     self.Y_all = torch.cat([self.Y_all.clone(), new_y_flat.detach().view(-1).squeeze(-1)], dim=0)
    #     self.Y_all = self.Y_all#.view(-1,1)
    #     #self.Y_all = torch.cat([self.Y_all, new_y_flat_standardized.detach()], dim=0)
    #     print ("Y_all.shape, new_y_flat.shape",self.Y_all.shape, new_y_flat.shape, new_y_flat.view(-1).squeeze(-1).shape)

    #     self.X_all = self.X_all#.detach()
    #     self.Y_all = self.Y_all#.detach()
    #     self.X_all = self.X_all.double()
    #     self.Y_all = self.Y_all.double()

    #     # Set the new training data for the model
    #     #self.Y_all_standardized = standardize(self.Y_all + 0.05 * torch.randn_like(self.Y_all))
    #     #self.Y_all.requires_grad = True
    #     likelihood = GaussianLikelihood()
    #     #print ("that",init_x_flat.shape, init_y.shape)
    #     #self.model = SingleTaskGP(init_x_flat, init_y, likelihood= likelihood, input_transform=intf, outcome_transform=Standardize(m=1))
    #     self.model.double()  # Ensure the model parameters are double
    #     # Reshape Y from [n] to [n, 1]
    #     intf = InputPerturbation(
    #         perturbation_set=draw_sobol_normal_samples(d=self.X_all.shape[-1], n=min ( num_samples,len (self.X_all)), **tkwargs) * STD_DEV,
    #         bounds=bounds_tensor,
    #         )
    #     self.model = SingleTaskGP(self.X_all, self.Y_all.clone().view(-1, 1), input_transform=intf, outcome_transform=Standardize(m=1))
    #     #self.model = SingleTaskGP(self.X_all, self.Y_all, input_transform=intf, outcome_transform=Standardize(m=1))
    #     self.model.set_train_data(inputs=self.X_all, targets=self.Y_all.clone().view(-1, 1), strict=False)

    #     # Optimize the model
    #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
    #     fit_gpytorch_model(mll)

    #     # Check for significant improvement with the best values
    #     # Calculate the average of best_Y
    #     average_best_Y = torch.mean(self.best_Y)

    #     # Check if the absolute difference between the average of best_Y and the best_value is less than the threshold
    #     # if abs(average_best_Y - self.best_value) < improvement_threshold and self.best_value!=0:
    #     #     self.start_cond = True
    #     #     continue
    #     # else:
    #     #     self.best_value = torch.max(self.best_Y)  # Update self.best_value to the new average
    #     #     break


    #     # Process the entire batch of best candidates

    #     #best_reshaped = candidates[best_indices].view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups).detach()
    #     #self.best_X =self.unnormalize(best_reshaped, self.bounds).detach().clone()
    #     self.best_X =unnormalize(self.best_X, bounds_tensor).detach().clone()#.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #     best_X_reshaped= self.best_X.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
    #     noise_tensor = self.gaussian_noise_subspace(best_X_reshaped)
    #     self.mu_prime_Batch = self.calculate_noisy_embeddings(self.group_embeddings, noise_tensor, True).to(self.device)

    #     del candidates, batch_candidates, batch_y, noise_tensor, new_y_flat, new_y_list
    #     torch.cuda.empty_cache()
    #     # self.best_X = self.best_X.to(self.device)
    #     # self.best_Y = self.best_Y.to(self.device)
    #     return self.best_X



#     def run_bayesian_optimization(self, actor, PLM, candid, num_samples=100, maxiter=20, improvement_threshold=0.0001, mixed_sampling_weight=0.2, num_restarts=6, raw_samples=100, batch_limit=8):
#         best_value = float('-inf')
#         self.samples = []
#         total_num_tasks = 1#self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups

#         print ("entering the function bo")
#         print_gpu_usage()
#         upsampleq=3
#         gc.collect()
#         torch.cuda.empty_cache()
#         device_cpu = torch.device("cpu")

#         while True:
#             bounds_tensor = torch.tensor([torch.min(self.bounds), torch.max(self.bounds)]).repeat(self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups, 1).double()

#             # Transpose the tensor to get the desired shape
#             bounds_tensor = bounds_tensor.T
#             bounds_tensor=bounds_tensor.to(device_cpu)
#             self.group_embeddings = self.calculate_embeddings(actor)
#             if self.start_cond == True:
#                 self.X_all = torch.empty(0, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()
#                 self.Y_all = torch.empty(0).double()

#                 self.X_all=self.X_all.to(device_cpu)

#                 init_x = self.initialize_data().double()  # Ensure init_x is double
#                 init_x=init_x.to(device_cpu)

#                 init_x_normalized = self.normalize(init_x, self.bounds)

#                 init_y = self.problem(init_x, actor, PLM).detach().double()  # Ensure init_y is double
#                 print(init_x.size(), init_y.size(), "final")
#                 #print (init_x,"init_x" ,init_y,"init_y" )
#                 init_y= standardize(init_y)
#                 #print ("dev",init_y.device)
#                 init_y=init_y.to(device_cpu )

#                 init_x_flat = init_x_normalized.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).detach().double()  # Ensure init_x_flat is double
#                 task_feature=init_x_flat.shape[-1] - 1
#                 #likelihood = MultitaskGaussianLikelihood(num_tasks=total_num_tasks)
#                 likelihood = GaussianLikelihood()
#                 print ()

#                 self.model = SingleTaskGP(init_x_flat, init_y,likelihood=likelihood)

#                 # Use the likelihood from the model
#                 #self.likelihood = self.model.likelihood


#                 #self.likelihood.double()

#                 #self.model = botorch.models.multitask.MultiTaskGP(init_x_flat, init_y,-1, likelihood)#MultiOutputGPModel(init_x_flat, init_y, total_num_tasks)
#                 self.model.double()  # Ensure the model parameters are double

#                 self.X_all = torch.cat([self.X_all, init_x_flat], dim=0)
#                 self.Y_all= self.Y_all.to(device_cpu)
#                 #print ("alldev",self.device,self.Y_all.device, init_y.device )
#                 self.Y_all = torch.cat([self.Y_all, init_y.detach().squeeze(-1)], dim=0)

#                 print ("sizes", init_x_flat.size(), init_y.size(), self.X_all.size(), self.Y_all.size())
#                 #self.model.set_
#                 mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

#                 fit_gpytorch_model(mll)
#                 self.best_value = best_value
#                 self.start_cond = False
#                 #posterior_transform = SelectFirstOutputTransform()

#             else:
#                 init_x = candid.double()  # Ensure candid is double
#                   # Ensure init_y is double
#                 # init_x_normalized = self.normalize(init_x, self.bounds)

#                 # init_x_flat = init_x_normalized.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()  # Ensure init_x_flat is double
#                 # print (init_x_flat.size(),"init_x_flat.size()")
#                 # task_feature=init_x_flat.shape[-1] - 1
#                 # #
#                 # init_y = self.problem(init_x_flat, actor, PLM).double()
#                 # init_y= standardize(init_y + 0.05 * torch.randn_like(init_y))
#                 # #init_y = standardize(init_y + 1e-1 * torch.randn_like(init_y))


#             d = init_x_flat.shape[-1]



#             #initial_candidates = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * torch.rand(num_samles, self.batch_size, d).double()
#             #initial_candidates = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * torch.rand( num_samples, self.batch_size , d).double()

#             #qMVE = qMaxValueEntropy(self.model, initial_candidates, posterior_transform=posterior_transform)

#             #print (bounds_tensor[0] , "yyy", (bounds_tensor[1] - bounds_tensor[0]), "xxxx")
#             #pareto_Y = init_y  # Simplification, use actual Pareto front here
#             #ref_point = pareto_Y.min(dim=0).values - 0.1  # Define a reference point lower than the Pareto front

# # Correctly initialize the sampler with sample_shape
#             print_gpu_usage()


#             #ref_point = infer_reference_point(init_y)
#             #print ("ref_point", ref_point, ref_point.size())


#             # Initialize the partitioning for EHVI
#             #partitioning = NondominatedPartitioning(ref_point=ref_point, Y=init_y)

#             # for name, param in self.model.named_parameters():
#             #   print(f"{name} requires_grad: {param.requires_grad}")
#             init_y.requires_grad = True
#             # Initialize the acquisition function
#             print ("before acq function")
#             print_gpu_usage()
#             # Put the model into training mode and find optimal model hyperparameters
#             # Use the best-fitting parameters
#             # self.model.mean_module = ConstantMean()
#             # self.model.covar_module = ScaleKernel(RBFKernel())
#             self.model.to(device_cpu )
#             self.model.likelihood.to(device_cpu)


#             # self.model.train()
#             # self.model.likelihood.train()

#             # Define the acquisition function
#             # self.model.eval()
#             # self.model.likelihood.eval()
#             print ("Y_all", self.Y_all.size(), self.Y_all)

#             # Initialize qNEI acquisition function
#                         # qKG = qKnowledgeGradient(
#             #     model=self.model,
#             #     #X_baseline=self.X_all,  # Updated baseline points
#             #     sampler=sampler,
#             #     num_fantasies=num_samples  # number of fantasies can be adjusted for exploration/exploitation tradeof
#             #     )

#             sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))

#             qNEI = qNoisyExpectedImprovement(
#                     model=self.model,
#                     X_baseline=self.X_all,  # Updated baseline points
#                     sampler=sampler,
#                                     )

#             # Optimize the acquisition function

#             #UCB = UpperConfidenceBound(model=self.model, beta=0.2)

#             candidates, acq_value = optimize_acqf(
#                 acq_function=qNEI,
#                 bounds=bounds_tensor,
#                 q=self.batch_size*upsampleq,
#                 num_restarts=num_restarts,
#                 raw_samples=raw_samples,  # Number of samples for initialization
#             )

#             # candids = []
#             # for _ in range(self.batch_size*upsampleq):
#             #     UCB = UpperConfidenceBound(model=self.model, beta=0.2)
#             #     candidate, acq_value = optimize_acqf(
#             #         acq_function=UCB,
#             #         bounds=bounds_tensor,
#             #         q=1,  # Generate one candidate at a time
#             #         num_restarts=num_restarts,
#             #         raw_samples=raw_samples,
#             #     )
#             #     candids.append(candidate)
#             # candidates= torch.cat(candids, dim=0)

#             candidates = candidates.detach()  # Detach once here
#             torch.cuda.empty_cache()
#             # Unnormalize candidates if your bounds were not [0, 1]
#             #proposed_points = unnormalize(candidates.detach(), self.bounds)
#             #proposed_points.requires_grad = False
#             #print("Proposed points:", proposed_points)

#             # proposed_points = unnormalize(candidates.detach(), bounds=bounds_tensor)
#             print ("after optimization.")
#             print_gpu_usage()

#             # for obj in gc.get_objects():
#             #   try:
#             #       # Check if it's a tensor on CUDA and not relevant
#             #       if torch.is_tensor(obj) and obj.is_cuda:
#             #           print (id(obj), "1")  # Move to CPU
#             #       # Check if it's an object containing a tensor on CUDA and not relevant
#             #       elif hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda :
#             #           print (id(obj), "2")  # Move the tensor within to CPU
#             #   except Exception as e:
#             #       pass

#             # Unnormalize candidates if your bounds were not [0, 1]

#             print ("yes!!")
#             new_y_values = []

#             #candidate_reshaped = candidates_flat.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
#             candidate_reshaped = candidates.view(self.batch_size*upsampleq,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
#             self.candidates =self.unnormalize(candidate_reshaped, self.bounds).detach().clone()
#             self.candidates.requires_grad_(False)
#             new_y_list = []

#             for i in range(upsampleq):
#                 # Selecting a batch of candidates
#                 batch_candidates = self.candidates[i * self.batch_size:(i + 1) * self.batch_size]

#                 # Process the batch and get results
#                 batch_y = self.problem(batch_candidates.detach(), actor, PLM).detach()

#                 # Append results to the list
#                 new_y_list.append(batch_y)

#             # Concatenate all the results along the first dimension
#             new_y_flat = torch.cat(new_y_list, dim=0)
#             #new_y_flat = self.problem(self.candidates.detach(), actor, PLM).detach()
#                         # Assuming new_y_flat contains the objective values for each candidate and you want to maximize these values
#             sorted_indices = torch.argsort(new_y_flat, descending=True)  # Change to ascending=False if you want to minimize

#             # Select the top batch_size candidates
#             candidates=candidates.to(device_cpu)
#             best_indices = sorted_indices[:self.batch_size]
#             best_indices=best_indices.to(device_cpu)


#             # Update best_X and best_Y with the best candidates and their corresponding y values
#             #print ("candidates",candidates)
#             self.best_X = candidates[best_indices]
#             print ("self.best_X.shape",self.best_X.shape)

#             new_y_flat = new_y_flat.to(device_cpu)
#             new_y_flat_standardized=standardize(new_y_flat+ 0.05 * torch.randn_like(new_y_flat))
#             self.best_Y = new_y_flat[best_indices]

#             # Detach and reshape for consistency
#             self.best_X = self.best_X#.detach().clone()
#             self.best_Y = self.best_Y.detach().clone().view(-1).squeeze(-1)

#             # Update X_all and Y_all with all candidates and their corresponding y values
#             self.X_all = torch.cat([self.X_all, candidates.detach()], dim=0)

#             self.Y_all = torch.cat([self.Y_all, new_y_flat_standardized.detach().view(-1).squeeze(-1)], dim=0)
#             print ("Y_all.shape, new_y_flat.shape",self.Y_all.shape, new_y_flat.shape, new_y_flat.view(-1).squeeze(-1).shape)

#             self.X_all = self.X_all#.detach()
#             self.Y_all = self.Y_all#.detach()
#             self.X_all = self.X_all.double()
#             self.Y_all = self.Y_all.double()

#             # Set the new training data for the model
#             #self.Y_all_standardized = standardize(self.Y_all + 0.05 * torch.randn_like(self.Y_all))
#             self.Y_all.requires_grad = True

#             self.model.set_train_data(inputs=self.X_all, targets=self.Y_all, strict=False)

#             # Optimize the model
#             mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
#             fit_gpytorch_model(mll)

#             # Check for significant improvement with the best values
#             # Calculate the average of best_Y
#             average_best_Y = torch.mean(self.best_Y)

#             # Check if the absolute difference between the average of best_Y and the best_value is less than the threshold
#             if abs(average_best_Y - self.best_value) < improvement_threshold and self.best_value!=0:
#                 self.start_cond = True
#                 continue
#             else:
#                 self.best_value = torch.max(self.best_Y)  # Update self.best_value to the new average
#                 break


#         # Process the entire batch of best candidates

#         best_reshaped = candidates[best_indices].view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups).detach()
#         self.best_X =self.unnormalize(best_reshaped, self.bounds).detach().clone()
#         noise_tensor = self.gaussian_noise_subspace(self.best_X)
#         self.mu_prime_Batch = self.calculate_noisy_embeddings(self.group_embeddings, noise_tensor, True).to(self.device)

#         del candidates, batch_candidates, batch_y, best_reshaped, noise_tensor, new_y_flat, new_y_list
#         torch.cuda.empty_cache()
#         # self.best_X = self.best_X.to(self.device)
#         # self.best_Y = self.best_Y.to(self.device)
#         return self.best_X


    def plot_reward_distribution(self):

        # Ensure Y_all is a tensor and then flatten it
        Y_all_flattened = self.Y_all.view(-1).squeeze(-1).flatten()

        # Convert each tensor element to a scalar
        scalar_rewards = [r.item() for r in Y_all_flattened]

        rewards_array = np.array(scalar_rewards)
        num_bins = int(np.sqrt(len(rewards_array)))
        #pdb.set_trace()
        # Create a histogram
        plt.hist(rewards_array, bins=num_bins , edgecolor='black')

        # Adding titles and labels
        plt.title('Distribution of Rewards')
        plt.xlabel('Reward Value')
        plt.ylabel('Frequency')
        plt.show()

        # Save the plot to a file
        plt.savefig('rewards_distribution.png', format='png', dpi=300)
        plt.close()

    def calculate_embedding(self,actor, sentence: str):
        #inputs = self.tokenizer(sentence, return_tensors='pt')
        with T.no_grad():
            #outputs = actor.forward(language_model, sentence)
            outputs = actor.forward_wte( sentence)#.squeeze(0)
            #print ("bb",outputs.shape)
            #print ("hi", outputs)
        return outputs


    def calculate_embeddings(self, actor):

        group_embeddings = []

        for group_pair_id, group_pair in enumerate(self.sampling_group_pairs):
            group_embeddings_pair = []

            for group_id, group in enumerate(group_pair):
                # Generate the sentence using the template
                sentence = self.template.format(group)

                # Tokenize the sentence
                sentence_tokens = actor.tokenizer.encode(sentence, return_tensors='pt').squeeze().tolist()
                sentence_token_str = actor.tokenizer.convert_ids_to_tokens(sentence_tokens)

                # Find the start and end index of the group in the sentence
                start_index = sentence.find(group)
                end_index = start_index + len(group)


                # Iterate over the tokenized sentence and find where the group starts and ends

                inputs = actor.tokenizer(sentence, return_tensors='pt')
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                group_token_id = actor.tokenizer.encode(" " + group)
                  #group_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
                group_start_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
                group_positions = [group_start_positions + i for i in range(len(group_token_id))]

                # Calculate the sentence embeddings and extract the group embeddings
                sentence_embedding = self.calculate_embedding(actor,  sentence)
                group_embedding = sentence_embedding[group_positions, :]#.squeeze()#(dim=-1)
                group_embedding.to(self.device)
                if group_embedding.size(0) != 2:
                    # Calculate the padding needed. In this case, we need one more row.
                    padding_needed = 2 - group_embedding.size(0)

                    # Create a tensor of zeros with the required padding size
                    padding = torch.zeros(padding_needed, group_embedding.size(1), device=self.device)

                    # Concatenate the original tensor with the padding
                    group_embedding = torch.cat([group_embedding, padding], dim=0)

                group_embeddings_pair.append(group_embedding)

            group_embeddings.append(group_embeddings_pair)

        # Convert group_embeddings to a tensor with the shape (group_pair_id, group_id, embedding)
        group_embeddings_tensor = T.stack([T.stack(pair) for pair in group_embeddings])

        return group_embeddings_tensor


    def calculate_noisy_embeddings(self, orig_embeddings: T.Tensor, noise: T.Tensor, checknoise=True):
        # Expand orig_embeddings to match the batch dimension of noise
        # orig_embeddings is expanded to (Batch_size, group_pair_id, group_id, embedding_size)
        expanded_orig_embeddings = orig_embeddings.unsqueeze(0).expand_as(noise).to(self.device)
        noise=noise.to(self.device)
        # Add the noise to the original embeddings
        # The broadcasting mechanism will handle the addition for each batch element
        if checknoise==True:
            noisy_embeddings = expanded_orig_embeddings + noise
        else:
            noisy_embeddings = expanded_orig_embeddings

        return noisy_embeddings


    def initialize_data(self):
        # Initialize the default_values tensor with an additional batch dimension
        self.default_values = torch.zeros((self.batch_size, self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups))

        for i in range(self.bias_subspace_dim):
            lower_bound = 0#self.bounds[0, i]
            upper_bound = 1#self.bounds[1, i]

            # Generate a new base_value for each i, to be used across all batches, j, and k
            base_value = lower_bound + (upper_bound - lower_bound) * torch.rand(1)
            #print ( "self.bounds", self.bounds)
            print ( "base_value", base_value)
            for b in range(self.batch_size):
                for j in range(self.num_groups_per_pair):
                    for k in range(self.num_groups):
                        # Assign the same base_value for all indexes of batch, j, k
                        self.default_values[b, i, j, k] = base_value.item()

        return self.default_values



    def sample(self, actor, PLM):

        #size = (self.num_groups, self.num_groups_per_pair, self.bias_subspace_dim)

        if self.start_cond==True:
            candidates=self.default_values
            #self.start_cond= False
        else:
            candidates= self.best_X

        # Run Bayesian optimization and get the noise mean value
        bo_results = self.run_bayesian_optimization(actor, PLM, candidates)



    def sample_batch(self, actor, PLM, checknoise=True):

        group_embeddings = self.calculate_embeddings(actor).to(self.device) #T.zeros((self.num_groups, self.num_groups_per_pair,self.num_tokens, self.embedding_dim))  # Initialize tensor for embeddings

        # for group_pair_id, group_pair in enumerate(self.sampling_group_pairs):
        #     for group_id, group in enumerate(group_pair):
        #         # Calculate embeddings for each group in the pair
        #         embedding = self.calculate_embeddings(LM)

        #         # Store embeddings in the tensor
        #         group_embeddings[group_pair_id, group_id, :,:] = embedding

        # Calculate noisy base embedding
        if checknoise==True:
          self.sample(actor, PLM)
          noisy_base_embedding = self.calculate_noisy_embeddings(group_embeddings.detach(), self.mu_prime_Batch.to(self.device), checknoise).to(self.device)

          return noisy_base_embedding, group_embeddings

        else:
          return group_embeddings

class ActorNetwork(nn.Module,metaclass=SingletonType):

    # no need for additional_layer, name
    def __init__(self, network, language_model,device,num_batches,lr,
         in_net=False, in_net_init_identity=False, out_net=False, out_net_init_identity=False, freeze_ln=False, freeze_pos=False,
                              freeze_wte=True, freeze_ff=True, freeze_attn=True, dup_lm_head=False, dup_lm_head_bias=False, chkpt_dir='/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/'):

        super(ActorNetwork, self).__init__()
        self.num_epochs = 1
        self.lr= lr
        self.current_epoch = -1  # Start from epoch ?
        self.tokenizer= language_model.tokenizer
        self.device = device
        self.num_batches = num_batches
        self.dup_lm_head= dup_lm_head
        self.checkpoint_file= chkpt_dir+'checkpoint2.pth'
        in_layer_sizes = []
        out_layer_sizes = []
        self.in_net= in_net
        self.out_net=out_net
        self.model= language_model.model.to(self.device)
        self.input_dim = language_model.model.config.n_embd
        dropout = 0.1
        orth_gain = 1.41
        in_net_init_identity = True
        self.freeze_ln=freeze_ln
        self.freeze_wte= freeze_wte
        self.freeze_pos=freeze_pos
        
        target_parameters = 0

        # for name, p in self.model.transformer.named_parameters():
        #     name = name.lower()

        #     size = p.size()
            # param_count = 1
            # for dimension in size:
            #     param_count *= dimension

            #total_parameters += param_count

#             if 'ln' in name or 'norm' in name:
#                 p.requires_grad = not self.freeze_ln
#             elif 'wpe' in name or 'position_embeddings' in name or 'pos_drop' in name:
#                 p.requires_grad = not self.freeze_pos
#                 #target_parameters += param_count
#             elif 'wte' in name:
#                 p.requires_grad = not self.freeze_wte
#             else:
#                 p.requires_grad = False

# #        self.optimizer =optim.Adam(self.model.parameters(), self.lr)
#                 #self.optimizer =optim.Adam(self.model.parameters(), lr=1e-4)
#         # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

#         # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches)

        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad], lr=self.lr
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches)



        # if in_net:
        #     in_layers = []
        #     last_output_size = self.input_dim
        #     self.model.in_net.requires_grad = True

        #     for size in in_layer_sizes:
        #         layer = nn.Linear(last_output_size, size)
        #         if orth_gain is not None:
        #             T.nn.init.orthogonal_(layer.weight, gain=orth_gain)
        #         layer.bias.data.zero_()

        #         in_layers.append(layer)
        #         in_layers.append(nn.ReLU())
        #         in_layers.append(nn.Dropout(dropout))
        #         last_output_size = size

        #     in_final_linear = nn.Linear(last_output_size, self.model.config.n_embd)
        #     # if orth_gain is not None:
        #     #     torch.nn.init.orthogonal_(in_final_linear.weight, gain=orth_gain)
        #     # in_final_linear.bias.data.zero_()

        #     # Initialize final_linear layer to identity transformation
        #     if in_net_init_identity:
        #         nn.init.eye_(in_final_linear.weight)
        #         in_final_linear.bias.data.zero_()

        #     in_layers.append(in_final_linear)
        #     in_layers.append(nn.Dropout(dropout))

        #     self.model.in_net = nn.Sequential(*in_layers)

        #     self.model.in_net.requires_grad = True

        # """
        # Initialize linear output layer
        # """
        # if out_net:
        #     output_dim = self.model.config.n_embd
        #     out_layers = []
        #     last_output_size = self.model.config.n_embd
        #     self.model.out_net.requires_grad = True

        #     for size in out_layer_sizes:
        #         out_layers.append(nn.Linear(last_output_size, size))
        #         out_layers.append(nn.ReLU())
        #         out_layers.append(nn.Dropout(dropout))
        #         last_output_size = size

        #     out_final_linear = nn.Linear(last_output_size, output_dim)

        #     if out_net_init_identity:
        #         nn.init.eye_(out_final_linear.weight)
        #         out_final_linear.bias.data.zero_()

        #     out_layers.append(out_final_linear)
        #     self.model.out_net = nn.Sequential(*out_layers)

        #     self.model.out_net.requires_grad = True

        # """
        # out layer on top of lm_head
        # """
        # # #
        # # out_net_top = nn.Linear(model.config.vocab_size, model.config.vocab_size)
        # # nn.init.eye_(out_net_top.weight)
        # # model.out_net_top = out_net_top
        # # model.out_net_top.requires_grad = True

        # # duplicated lm_head
        # if dup_lm_head:
        #     lm_head_new = nn.Linear(self.model.config.n_embd,
        #                             self.model.config.vocab_size, bias=dup_lm_head_bias)
        #     lm_head_new.weight = T.nn.Parameter(
        #         self.model.lm_head.weight.data.detach().clone(), requires_grad=True)
        #     # lm_head_new.bias.data.zero_()
        #     self.model.lm_head_new = lm_head_new
        #     self.model.lm_head_new.requires_grad = True
        #     for param in self.model.lm_head_new.parameters():
        #         param.requires_grad = True

        """
        Freeze transformer layers
        """

        #total_parameters = 0
#         target_parameters = 0

#         for name, p in self.model.transformer.named_parameters():
#             name = name.lower()

#             size = p.size()
#             # param_count = 1
#             # for dimension in size:
#             #     param_count *= dimension

#             #total_parameters += param_count

#             if 'ln' in name or 'norm' in name:
#                 p.requires_grad = not freeze_ln
#             elif 'wpe' in name or 'position_embeddings' in name or 'pos_drop' in name:
#                 p.requires_grad = not freeze_pos
#                 #target_parameters += param_count
#             elif 'wte' in name:
#                 p.requires_grad = not freeze_wte
#             else:
#                 p.requires_grad = False

# #        self.optimizer =optim.Adam(self.model.parameters(), self.lr)
#                 #self.optimizer =optim.Adam(self.model.parameters(), lr=1e-4)
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches)



    def increment_epoch(self):
        self.current_epoch += 1
        self.update_trainable_layers()

    def update_trainable_layers(self):
        
        # transformer_blocks = self.model.transformer.h  # Accessing the transformer blocks
        # # Reversing the order of transformer blocks
        # all_layers = list(reversed(transformer_blocks))
        # # Determine number of layers to unfreeze based on the current epoch
        # num_layers_to_unfreeze = min(self.current_epoch, len(all_layers))
        # print ("num_layers_to_unfreeze",num_layers_to_unfreeze)
        # del transformer_blocks

        
        # # Loop through each layer
        # if num_layers_to_unfreeze <4:
        #     for i, layer in enumerate(all_layers):
        #         # Enable gradients for the first num_layers
        #         if i < num_layers_to_unfreeze:
        #             for param in layer.parameters():
        #                 param.requires_grad = True
        #         else:
        #             for param in layer.parameters():
        #                 param.requires_grad = False

        # else:
        
        #     for i, layer in enumerate(all_layers):
        #         for param in layer.parameters():
        #             param.requires_grad = True
        


        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches)
        # del all_layers

       
        num_transformer_blocks = len(self.model.transformer.h)  # Total number of transformer blocks in the model

        total_layers = len(self.model.transformer.h)  # Total number of transformer blocks in the model
        
        # Define the start layer from which you want to unfreeze the layer normalization parameters
        #start_unfreeze_layer = total_layers - 4

        start_unfreeze_layer = max(0, total_layers - 4 - self.current_epoch)
        print ("start_unfreeze_layer",start_unfreeze_layer)

        for name, param in self.model.named_parameters():
            # Initially freeze all parameters
                param.requires_grad = False
                
            # For the first 3 epochs, only train the lm_head (ln_f) parameters
            # if self.current_epoch == :
            #     if 'ln_f.weight' in name or 'ln_f.bias' in name:
            #         param.requires_grad = True
            # else:
            #     # Logic for epochs >= 3 where you decide which parameters to train
                # Example logic based on your initial setup

                # Using a regex to match layer numbers and parameter names for layer normalization
                #match = re.match(r'transformer.h\.(\d+)\.(ln_[12]\.weight|ln_[12]\.bias)', name)
                match = re.match(r'transformer\.h\.(\d+)\.(ln_[12]\.weight|ln_[12]\.bias)', name)

                if self.current_epoch<2:
                    if 'ln_f.weight' in name or 'ln_f.bias' in name:
                        param.requires_grad = True

                    
                    break
                    
                elif match and self.current_epoch>2:
                    layer_index = int(match.group(1))  # Convert captured layer index to integer
                    print ("layer_index",layer_index)
                    print ("match", match)
                    # Calculate the layer to start unfreezing from, based on the current epoch
                    #if layer_index >= start_unfreeze_layer:
                    param.requires_grad = True  
                # if 'ln' in name or 'norm' in name:
                #     param.requires_grad = not self.freeze_ln

                if 'wpe' in name or 'position_embeddings' in name or 'pos_drop' in name:
                    param.requires_grad = not self.freeze_pos

                if 'wte' in name:  # Token embeddings
                    param.requires_grad = not self.freeze_wte
                
                if self.current_epoch >5 and 'h.11' in name:
                    param.requires_grad =True
                    
                if self.current_epoch >6 and 'h.10' in name:
                    param.requires_grad =True

                if self.current_epoch >7 and 'h.9' in name:
                    param.requires_grad =True


                # You could add more specific conditions here based on your model's structure and training needs

        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches)

        base_params = [param for name, param in self.model.named_parameters() if 'h.11' not in name and param.requires_grad]
        lm_head_params = [param for name, param in self.model.named_parameters() if 'ln.f' in name and param.requires_grad]

        # Now, set up the optimizer with different learning rates
        if base_params or lm_head_params:
            self.optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': self.lr},  # Standard learning rate for base model parameters
                {'params': lm_head_params, 'lr': self.lr*0.1}  # Adjusted learning rate for LM head parameters
            ])
        else:
            raise ValueError("No parameters with requires_grad=True. Check your model's parameter setup.")

        # Setup the scheduler with the optimizer
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches)


    def forward_text(self, text): # this forward's output is aka state

        #device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')


        inputs= self.tokenizer(text, return_tensors='pt')
        self.model = self.model.to(self.device)

        # Move the inputs to the correct device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Ensure the model is on the correct device
        #self.model.eval()
        outputs = self.model(**inputs)


        return outputs.hidden_states[-1].squeeze(0)


    def forward_action(self, action):
        self.model = self.model.to(self.device)

        #print("forward_action")
        #print("Action shape:", action.shape)  # Debugging line


        outputs = self.model.transformer(inputs_embeds=action)


        #print ("outputs",outputs)
        return outputs#.hidden_states[-1].squeeze(0)


    def forward_group(self, text, group_idx):

        inputs = self.tokenizer(text, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        #self.model.eval()
        outputs = self.model(**inputs)

        # Get the last hidden state
        last_hidden_states = outputs.hidden_states[-1].squeeze(0)  # Last layer hidden states

        if isinstance(group_idx, int):
            group_x = last_hidden_states.squeeze()[group_idx]

        elif isinstance(group_idx, list):
            group_x = T.mean(last_hidden_states.squeeze()[group_idx[0]:group_idx[-1]+1], dim=0)
        else:
            raise TypeError("group_idx must be either an integer or a list/tuple of integers.")

        return group_x

    def forward_wte(self, sentence):

        inputs = self.tokenizer(sentence, return_tensors="pt")

        # Get the token ids and remove the batch dimension
        input_ids = inputs['input_ids'].squeeze().to(self.device)


        # Get the word token embeddings (WTE)
        with T.no_grad():
            embeddings = self.model.transformer.wte(input_ids)

        return embeddings

    def forward_logits(self, sentence):

        inputs = self.tokenizer(sentence, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        self.model = self.model.to(self.device)
        #self.model.eval()
        with T.no_grad():
            outputs = self.model(input_ids)

        # GPT-2 directly outputs logits as part of its forward pass
        logits = outputs.logits
        logits = logits.squeeze(0)  # Remove the batch dimension if necessary

        return logits


    # def forward_state(self, filled_template, group_index, noisy_group_embedding):

    #     # Obtain the input embeddings for the entire sentence
    #     input_embeddings = self.forward_wte(filled_template)

    #     #print ("input_embeddings",input_embeddings.shape)
    #     #?original_group_embedding = input_embeddings[:, group_index, :]
    #     original_group_embedding = input_embeddings[ group_index, :]
    #     #noisy_group_embedding= noisy_group_embedding#.squeeze(0)
    #     #noisy_group_embedding= noisy_group_embedding.unsqueeze(0)
    #     # Replace the original embedding of the 'group' token with the noisy one

    #     #noisy_group_embedding = noisy_group_embedding.expand(input_embeddings.size(0),-1)
    #     #print ("noisy_group_embedding.shape",noisy_group_embedding.shape)
    #     input_embeddings[ group_index, :] = noisy_group_embedding


    #     # Pass the modified embeddings through the model
    #     outputs = self.model(inputs_embeds=input_embeddings)
    #     hidden_states = outputs.hidden_states

    #     # If you need the last layer's hidden states specifically
    #     last_hidden_states = hidden_states[-1]

    #     #print ("outputs.last_hidden_state.shape",last_hidden_states.shape)
    #     # Extract the final hidden state for the 'group' token
    #     group_hidden_state = last_hidden_states[ group_index, :]
    #     #print ("group_hidden_state.squeeze(0)" ,group_hidden_state.shape)
    #     return group_hidden_state # Remove the batch dimension and token if necessary

    # def forward(
    #     self,
    #     input_ids=None,
    #     past_key_values=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     encoder_hidden_states=None,
    #     encoder_attention_mask=None,
    #     labels=None,
    #     use_cache=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     return_dict=None,
    #     **kwargs
    # ):
    #     r"""
    #     labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
    #         Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
    #         ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
    #         ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
    #     """
    #     return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

    #     # Convert from input ids to word embeddings so that we can apply a linear layer
    #     x = self.model.transformer.wte(input_ids)

    #     if input_ids is not None:
    #         input_ids = input_ids.to(self.device)
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.to(self.device)
    #     if token_type_ids is not None:
    #         token_type_ids = token_type_ids.to(self.device)
    #     if position_ids is not None:
    #         position_ids = position_ids.to(self.device)
    #     if encoder_attention_mask is not None:
    #         encoder_attention_mask = encoder_attention_mask.to(self.device)
    #     if labels is not None:
    #         labels = labels.to(self.device)

    #     # Convert from input ids to word embeddings and apply a linear layer if exists
    #     x = self.model.transformer.wte(input_ids).to(self.device)
    #     try:
    #         x = self.model.in_net(x)
    #     except AttributeError:
    #         pass


    #     try:
    #         x = self.model.in_net(x)
    #     except AttributeError:
    #         pass


    #     transformer_outputs = self.model.transformer(
    #         inputs_embeds=x,
    #         past_key_values=past_key_values,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         encoder_hidden_states=encoder_hidden_states,
    #         encoder_attention_mask=encoder_attention_mask,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #         **kwargs
    #     )
    #     hidden_states = transformer_outputs[0]

    #     # # Set device for model parallelism
    #     # if self.model_parallel:
    #     #     T.cuda.set_device(self.transformer.first_device)
    #     #     hidden_states = hidden_states.to(self.model.lm_head.weight.device)

    #     try:
    #         hidden_states = self.model.out_net(hidden_states)
    #     except AttributeError:
    #         pass

    #     try:
    #         lm_logits = self.model.lm_head_new(hidden_states)
    #     except AttributeError:
    #         print ("lm head new not found")
    #         lm_logits = self.model.lm_head(hidden_states)

    #     # # TODO
    #     # lm_logits = self.out_net_top(lm_logits)

    #     loss = None
    #     if labels is not None:
    #         # Shift so that tokens < n predict n
    #         shift_logits = lm_logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         # Flatten the tokens
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(
    #             shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # if not return_dict:
        #     output = (lm_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

    #     return CausalLMOutputWithCrossAttentions(
    #         loss=loss,
    #         logits=lm_logits,
    #         past_key_values=transformer_outputs.past_key_values,
    #         hidden_states=transformer_outputs.hidden_states,
    #         attentions=transformer_outputs.attentions,
    #         cross_attentions=transformer_outputs.cross_attentions,
    #     )

    def resize_token_embeddings(self, new_num_tokens=None):
        if new_num_tokens is None:
            new_num_tokens = len(self.tokenizer)
        self.core_model.resize_token_embeddings(new_num_tokens)

    # def save_checkpoint(self): # this saved checkpoint stores the debiased finetuned PLM
    #     drive.mount('/content/drive')

    #     # Path to the folder in Google Drive where the checkpoint will be saved
    #     drive_folder = "/content/drive/My Drive/ActorModelCheckpoints"
    #     checkpoint_filename = "checkpoint.pth"  # Name of the checkpoint file
    #     checkpoint_file = os.path.join(drive_folder, checkpoint_filename)

    #     # Prepare the dictionary to be saved
    #     save_dict = {
    #         'model_state_dict': self.model.state_dict(),
    #         # Add other state dicts as needed
    #     }
    #     if hasattr(self.model, 'in_net'):
    #         save_dict['in_net_state_dict'] = self.model.in_net.state_dict()
    #     if hasattr(self.model, 'out_net'):
    #         save_dict['out_net_state_dict'] = self.model.out_net.state_dict()
    #     if hasattr(self.model, 'lm_head_new'):
    #         save_dict['lm_head_new_state_dict'] = self.model.lm_head_new.state_dict()

    #     # Create the folder if it does not exist
    #     if not os.path.exists(drive_folder):
    #         os.makedirs(drive_folder)
    #     print('... saving checkpoint ...')
    #     T.save(save_dict, checkpoint_file)

    #     # Debugging: Load immediately to check
    #     checkpoint_debug = T.load(checkpoint_file)
    #     #print("Loaded checkpoint keys:", checkpoint_debug.keys())
    def save_checkpoint(self, checkpoint_name="checkpoint2.pth"):
        # Mount Google Drive (specific to Google Colab)
        #drive.mount('/content/drive')

        # Specify the folder in Google Drive to save the checkpoint
        drive_folder = "/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/"
        checkpoint_file = os.path.join(drive_folder, checkpoint_name)

        # Ensure the folder exists
        if not os.path.exists(drive_folder):
            os.makedirs(drive_folder)

        # Prepare the checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            # Include any additional information or models if necessary
        }

        # Save the checkpoint
        print('Saving checkpoint to:', checkpoint_file)
        T.save(checkpoint, checkpoint_file)

        # Optional: Confirm the file has been saved
        if os.path.isfile(checkpoint_file):
            print('Checkpoint saved successfully.')
        else:
            print('Error: Checkpoint saving failed.')

        del checkpoint
# Example usage

    # def save_checkpoint(self): # this saved checkpoint stores the debiased finetuned PLM

    #     save_dict = {
    #     'model_state_dict': self.state_dict(),
    #     #'transformer_state_dict': self.transformer.state_dict()
    # }

    #     if hasattr(self.model, 'in_net'):
    #         save_dict['in_net_state_dict'] = self.model.in_net.state_dict()
    #     if hasattr(self.model, 'out_net'):
    #         save_dict['out_net_state_dict'] = self.model.out_net.state_dict()
    #     if hasattr(self.model, 'lm_head_new'):
    #         save_dict['lm_head_new_state_dict'] = self.model.lm_head_new.state_dict()

    # # Save the dictionary to a file

    #     directory = os.path.dirname(self.checkpoint_file)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     print('... saving checkpoint ...')
    #     T.save(save_dict, self.checkpoint_file)

    #     # Debugging: Load immediately to check
    #     checkpoint_debug = T.load(self.checkpoint_file)
    #     #print("hiii", checkpoint_debug.keys())




    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.model.checkpoint_file, device=self.device))




class CoVWeighting:
    def __init__(self, num_losses, device, decay_factor=0.25):
        self.num_losses = num_losses
        self.device = device
        self.decay_factor = decay_factor

        # Initialize statistics for losses (L) and loss ratios (l) as lists to handle batches
        self.running_mean_L = [torch.zeros(1, device=device) for _ in range(num_losses)]
        self.running_var_L = [torch.zeros(1, device=device) for _ in range(num_losses)]
        self.running_mean_l = [torch.zeros(1, device=device) for _ in range(num_losses)]
        self.running_var_l = [torch.zeros(1, device=device) for _ in range(num_losses)]

        self.iteration = 0

    def update_statistics(self, losses):
        """
        Update statistics for a batch of losses.

        :param losses: Tensor of shape [num_losses, batch_size], each row represents losses of the same type across the batch.
        """
        self.iteration += 1
        decay = self.decay_factor

        for i in range(self.num_losses):
            loss = losses[i]  # Shape: [batch_size]

            if self.iteration == 1:
                self.running_mean_L[i] = loss.mean().unsqueeze(0)
                self.running_mean_l[i] = torch.ones(1, device=self.device)  # Initialize to 1 to avoid division by zero
            else:
                # Update running means for L and loss ratios l
                self.running_mean_L[i] = decay * self.running_mean_L[i] + (1 - decay) * loss.mean().unsqueeze(0)
                loss_ratio = loss / self.running_mean_L[i]
                self.running_mean_l[i] = decay * self.running_mean_l[i] + (1 - decay) * loss_ratio.mean().unsqueeze(0)

                # Update variances for L and l
                delta_L = loss - self.running_mean_L[i]
                delta_l = loss_ratio - self.running_mean_l[i]
                self.running_var_L[i] = decay * self.running_var_L[i] + (1 - decay) * (delta_L ** 2).mean().unsqueeze(0)
                self.running_var_l[i] = decay * self.running_var_l[i] + (1 - decay) * (delta_l ** 2).mean().unsqueeze(0)

    def compute_weights(self):
        # Compute CoV for each loss and loss ratio
        cov_l = torch.tensor([torch.sqrt(var) / (mean + 1e-8) for var, mean in zip(self.running_var_l, self.running_mean_l)], device=self.device)

        # Calculate weights inversely proportional to CoV
        weights = 1 / (cov_l + 1e-8)
        normalized_weights = weights / weights.sum()

        return normalized_weights

    def adjust_loss_weights(self, losses):
        """
        Adjust loss weights based on a batch of losses for each loss objective.

        :param losses: Tensor of shape [num_losses, batch_size]
        :returns: Weighted sum of losses, Weights used for each loss objective
        """
        self.update_statistics(losses)
        weights = self.compute_weights()
        target_device = losses.device if isinstance(losses, torch.Tensor) else losses[0].device
        weights = weights.to(target_device)

        # Apply weights to each loss and sum over all losses for the batch
        weighted_losses = torch.sum(losses * weights[:, None], dim=0)  # Summing over losses, broadcasting weights

        return weighted_losses.mean(), weights  # Returning mean loss over the batch and the weights



class Reward(object):



    def __init__(self, Bigbatch_size, batch_size, PLM, lambda1 , sampling_group_pairs, traits, device ):
        self.device = device # T.device("cuda")
        self.lambda1  = lambda1
        self.num_iterations=0
        #self.tau = tau
        self.batch_size = batch_size
        self.Bigbatch_size = Bigbatch_size
        self.garbage_dict= {}
        self.garbage_dict_count=0
        self.num_tokens=2
        self.reward= None

        self.sampling_group_pairs=sampling_group_pairs
        self.traits= traits
        self.embedding_dim= PLM.d_model
        self.sampling_group_pairs=sampling_group_pairs #[ ('man', 'woman'),('stepfather', 'stepmother')]
        self.num_groups= len(self.sampling_group_pairs)
        self.num_groups_per_pair = len(self.sampling_group_pairs[0])  # Assuming each tuple in group_pairs has the same number of elements
        self.template = "The {} is "
        self.prime1,self.primeB= self.calculate_prime(PLM)
        self.NoiseE=[]
        self.klp=[]
        self.count = 0
        self.logits_count=0
        self.top_count=0
        self.neutral_count=0
        self.num_epochs=0
        self.device_cpu = torch.device("cpu")
        self.CoVWeighting=CoVWeighting(2,self.device_cpu) 

    def calculate_reward(self, mu_prime, LM, PLM, lambda1, weighted=False, both= False):

                #mu_prime, _= self.noise.sample_batch (LM)
        batchNoiseEval= self.NoiseEval( LM, mu_prime, self.sampling_group_pairs, self.traits, None, "The <group> is <mask>")
        
               #pdb.set_trace()
        if batchNoiseEval.size()[0] == self.batch_size:
            prime=self.prime1

        else:
            prime= self.primeB


        klG1, klG2= self.kl_perf( LM, PLM, mu_prime, prime, self.sampling_group_pairs, "The <group> is powerful")

        fairness_coefficient=20
        
        epsilon = 1e-6
        
        #print("combined_reward",combined_reward)
        self.NoiseE.append(batchNoiseEval.mean(0))
        self.klp.append((klG1 + klG2).mean(0)*lambda1)
        print ("batchNoiseEvalmagnified ,", batchNoiseEval ,"lambda1*(klG1+ klG2):",  lambda1*(klG1+ klG2))
        print ("batchNoiseEvalmagnified ,", batchNoiseEval)
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = batchNoiseEval
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = klG1
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = klG2

                
        # Assuming performance_loss and fairness_loss are already defined
        
        FAIRNESS_PRIORITY_ratio = 0.1  # Threshold for when to prioritize fairness
        performance_coefficient = 1
        # Condition to check if performance is satisfactory or if fairness needs prioritization
        

        # if self.num_iterations>60:
        #     while torch.min(batchNoiseEval*fairness_coefficient / (-lambda1 * performance_coefficient*(klG1 + klG2))) < FAIRNESS_PRIORITY_ratio:
        #         # Adjust the performance coefficient to prioritize fairness
        #         performance_coefficient *= 0.98  # Example: reduce by 10%

        # # Recalculate the dynamic loss with the updated coefficient

        # combined_reward = batchNoiseEval*fairness_coefficient - lambda1*performance_coefficient * (klG1 + klG2)  # Example of combining losses
        losses_tensor= torch.tensor([torch.mean(batchNoiseEval), torch.mean (-lambda1* (klG1 + klG2))])
        _, weights = self.CoVWeighting.adjust_loss_weights(losses_tensor)
        
        adjusted_reward= batchNoiseEval*weights[0] - lambda1* (klG1 + klG2)*weights[1]
        

        # if weighted==True:
        #     weights = 1 / (torch.abs(batchNoiseEval) + epsilon)

        #     # Normalize weights to sum to 1
        #     weights_normalized = weights / torch.sum(weights)

        #     # Compute the weighted average of the combined objective
        #     weighted_average = weights_normalized * combined_reward


        #     print("combined_reward",weighted_average)
        #     self.garbage_dict_count += 1
        #     self.garbage_dict[self.garbage_dict_count] = combined_reward
        #     self.garbage_dict_count += 1
        #     self.garbage_dict[self.garbage_dict_count] = weights_normalized        
        #     self.garbage_dict_count += 1
        #     self.garbage_dict[self.garbage_dict_count] = weighted_average        
        
        #     return weighted_average
        # # else:
        # self.garbage_dict_count += 1
        # self.garbage_dict[self.garbage_dict_count] = combined_reward
        
        if both == False:
            return adjusted_reward
        else:
            return torch.stack((batchNoiseEval*weights[0], - lambda1* (klG1 + klG2)*weights[1]+ batchNoiseEval*weights[0]), dim=1)

        #return combined_reward
        #return batchNoiseEval*20

    def calculate_embedding(self,actor, sentence: str):
        #inputs = self.tokenizer(sentence, return_tensors='pt')
        with T.no_grad():
            #outputs = actor.forward(language_model, sentence)
            outputs = actor.forward_wte( sentence)#.squeeze(0)
            #print ("bb",outputs.shape)
            #print ("hi", outputs)
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = outputs
        return outputs


    def calculate_embeddings(self, actor):

        group_embeddings = []

        for group_pair_id, group_pair in enumerate(self.sampling_group_pairs):
            group_embeddings_pair = []

            for group_id, group in enumerate(group_pair):
                # Generate the sentence using the template
                sentence = self.template.format(group)

                # Tokenize the sentence
                sentence_tokens = actor.tokenizer.encode(sentence, return_tensors='pt').squeeze().tolist()
                sentence_token_str = actor.tokenizer.convert_ids_to_tokens(sentence_tokens)

                # Find the start and end index of the group in the sentence
                start_index = sentence.find(group)
                end_index = start_index + len(group)

                # Initialize the group indices
                group_indices = []

                # Iterate over the tokenized sentence and find where the group starts and ends

                inputs = actor.tokenizer(sentence, return_tensors='pt')
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                group_token_id = actor.tokenizer.encode(" " + group)
                  #group_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
                group_start_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
                group_positions = [group_start_positions + i for i in range(len(group_token_id))]

                # Calculate the sentence embeddings and extract the group embeddings
                sentence_embedding = self.calculate_embedding(actor,  sentence)
                group_embedding = sentence_embedding[group_positions, :]#.squeeze()#(dim=-1)
                group_embedding = group_embedding.to(self.device)
                if group_embedding.size(0) != 2:
                    # Calculate the padding needed. In this case, we need one more row.
                    padding_needed = 2 - group_embedding.size(0)

                    # Create a tensor of zeros with the required padding size
                    padding = torch.zeros(padding_needed, group_embedding.size(1))
                    group_embedding = group_embedding.to(self.device)
                    padding = padding.to(self.device)

                    # Concatenate the original tensor with the padding
                    group_embedding = torch.cat([group_embedding, padding], dim=0)
                    group_embedding = group_embedding.to(self.device)

                group_embeddings_pair.append(group_embedding)

            group_embeddings.append(group_embeddings_pair)

        # Convert group_embeddings to a tensor with the shape (group_pair_id, group_id, embedding)
        group_embeddings_tensor = T.stack([T.stack(pair) for pair in group_embeddings])

        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = group_embeddings
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = group_embeddings_pair
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = group_embeddings_tensor
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = sentence_embedding
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = inputs
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = sentence_tokens
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = sentence_token_str
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = group_token_id


        del group_embeddings
        del group_embeddings_pair
        del group_token_id
        del group_start_positions
        del group_positions
        del sentence_embedding

        return group_embeddings_tensor



    def calculate_prime(self, PLM):

        group_embeddings = T.zeros((self.num_groups, self.num_groups_per_pair, self.embedding_dim))  # Initialize tensor for embeddings

        # for group_pair_id, group_pair in enumerate(self.sampling_group_pairs):
        #     for group_id, group in enumerate(group_pair):
        #         # Calculate embeddings for each group in the pair

        #         # Store embeddings in the tensor
        #         group_embeddings[group_pair_id, group_id, :] = embedding
        group_embeddings = self.calculate_embeddings(PLM)
        #pdb.set_trace()

        primeB = group_embeddings.unsqueeze(0).expand(self.Bigbatch_size, -1, -1,-1,-1)

        prime1 = group_embeddings.unsqueeze(0).expand((1,-1,-1,-1,-1))
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = group_embeddings
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = prime1
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = primeB
        return prime1, primeB

    # SAME AS LOG PROB, WE CAN NOT PREDICT FORWARD WITH A VERY LOW LOGIT AN DTHEN ADD THIS, THE REUSLT REMAINS NEGLIGIBLE.
    def get_neutral_score(self, LM, tmplt, trait):# here you should predict multi token based on a sentence that you already added to the current trait token.
        selected_gpu = sys.argv[1]

        device = torch.device(f'cuda:{selected_gpu}')

        self.count += 1
        if self.neutral_count % 500 == 0:
          gc.collect()
          self.neutral_count = 0


        # Tokenize the sentence template without the mask token and the trait
        sentence = tmplt.replace('<mask>', '') #+ trait
        print ("sentence",sentence)

        inputs = LM.tokenizer(sentence, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Tokenize the trait and get its token IDs, excluding special tokens like [CLS] and [SEP]
        trait_ids = list(LM.tokenizer.encode(" "+trait))
        trait_ids= [trait_ids[0]]
        # Calculate the position where the trait starts in the input sequence
        trait_start_pos = inputs['input_ids'].size(1) - len(trait_ids)-1

        # Initialize log probability
        logit_sum = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        inputs=inputs.to(device)
        # Predict the probability of each token in the trait
        with torch.no_grad():
            outputs = LM.model(**inputs)
            logits = outputs["logits"]
        #outputs= self.to_cpu(outputs)


        print ("logits.size()",logits.size())
        #logits=logits.to(torch.device("cpu"))
        #pdb.set_trace()
        # Iterate over the trait_ids and sum their corresponding logits
        for i, trait_id in enumerate(trait_ids):
            #token_logits = logits[0, trait_start_pos + i, :]  # Get the logits for the current token position
            token_logits = logits[0, -1, :]
            trait_logit = token_logits[trait_id]  # Get the logit for the current trait_id
            logit_sum = logit_sum + trait_logit  # Sum the logits, maintaining the computational graph
            print ("trait_logit",trait_logit , trait_id , trait_ids , trait)
        # Ensure logit_sum is a single-value tensor; no need for view() here as it's already a single value
        print ("neutral score", logit_sum)
        #outputs=outputs.detach()
        #inputs=inputs.to(torch.device("cpu"))#detach()
        #logits=logits.to(torch.device("cpu"))#detach()

        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = outputs
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = inputs
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = logits
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = token_logits
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = trait_logit
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = trait_ids
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = trait_start_pos
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = logit_sum
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = trait
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = sentence


        del trait_ids
        del trait_id
        del token_logits
        del outputs
        del inputs
        del trait_start_pos

        return logit_sum



    # def get_neutral_score(self, LM, tmplt, trait):
    #     device = torch.device('cuda')

    #     # Tokenize the sentence template without the mask token and the trait
    #     sentence = tmplt.replace('<mask>', '') + trait
    #     inputs = LM.tokenizer(sentence, return_tensors='pt')
    #     inputs = {key: value.to(device) for key, value in inputs.items()}

    #     # Tokenize the trait and get its token IDs, excluding special tokens like [CLS] and [SEP]
    #     trait_ids = list(LM.tokenizer.encode(" "+trait))
    #     trait_ids_tensor = torch.tensor(trait_ids, device=device)

    #     # Calculate the position where the trait starts in the input sequence
    #     trait_start_pos = inputs['input_ids'].size(1) - len(trait_ids)

    #     # Initialize log probability
    #     logit_sum = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

    #     # Predict the probability of each token in the trait
    #     for i in range(len(trait_ids)):
    #         # Slice the input to include only up to the current token position
    #         current_inputs = {key: value[:, :trait_start_pos+i] for key, value in inputs.items()}

    #         with torch.no_grad():
    #           outputs = LM.model(**current_inputs)
    #         logits = outputs.logits

    #       # Iterate over the trait_ids and sum their corresponding logits

    #         current_token_logits = logits[0, -1, :]  # Use the logits of the last token in the sequence
    #         trait_logit = current_token_logits[trait_ids[i]]  # Get the logit for the current trait_id
    #         logit_sum = logit_sum + trait_logit  # Sum the logits, maintaining the computational graph

    #     del trait_ids_tensor
    #     del trait_ids
    #     del current_inputs
    #     del current_token_logits
    #     del trait_logit
    #     del outputs
    #     del logits
    #     del inputs
    #     del trait_start_pos

    #     # Ensure logit_sum is a single-value tensor; no need for view() here as it's already a single value
    #     print ("neutral score",logit_sum)
    #     return logit_sum#.unsqueeze(0)


    # def get_log_prob(self, LM, mu_prime_g, inputs, group, trait, PLM=None):


    #   device = self.device

    #   # Directly use inputs_embeds without detaching to maintain gradient flow
    #   inputs_embeds = LM.model.transformer.wte(inputs['input_ids'])

    #   # Calculate positions for 'group' tokens
    #   group_token_id = LM.tokenizer.encode(" " + group, add_special_tokens=False)
    #   group_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]

    #   # Prepare mu_prime_g for embedding replacement
    #   embedding_size = LM.model.config.n_embd
    #   mu_prime_g = mu_prime_g.view(-1, embedding_size).to(device)  # Ensure shape compatibility

    #   # Replace embeddings for 'group' positions directly, maintaining gradient flow
    #   for i, pos in enumerate(group_positions):
    #       inputs_embeds[0, pos, :] = mu_prime_g[i, :]

    #   # Extend inputs_embeds to accommodate trait prediction if necessary
    #   trait_ids = LM.tokenizer.encode(" " + trait, add_special_tokens=False)
    #   required_length = inputs_embeds.size(1) + len(trait_ids) - inputs['input_ids'].size(1)
    #   if inputs_embeds.size(1) < required_length:
    #       extension = torch.zeros((1, required_length - inputs_embeds.size(1), embedding_size), device=device)
    #       inputs_embeds = torch.cat([inputs_embeds, extension], dim=1)

    #   # Calculate logits with extended inputs_embeds, ensuring return_dict=True for easy access
    #   outputs = LM.model(inputs_embeds=inputs_embeds, return_dict=True)
    #   logits = outputs.logits

    #   # Compute log probabilities for each trait_id
    #   log_probs = []
    #   start_pos = inputs['input_ids'].size(1)  # Adjust if necessary for your specific use case
    #   for i, trait_id in enumerate(trait_ids):
    #       token_logits = logits[0, start_pos + i, :]
    #       token_prob = token_logits[trait_id]
    #       log_probs.append(token_prob)

    #   # Sum the log probabilities to get a single scalar for backpropagation
    #   total_log_prob = torch.sum(torch.stack(log_probs))

    #   return total_log_prob
    def calculate_logits(self, LM, tmplt, mu_prime_g, inputs, group):


        selected_gpu = sys.argv[1]

        device = torch.device(f'cuda:{selected_gpu}')


        
        self.logits_count += 1
        if self.logits_count % 50 == 0:
            gc.collect()
            self.logits_count = 0

        # input=inputs.to(self.device)
        with torch.no_grad():
            inputs_embeds = LM.model.transformer.wte(inputs['input_ids'])

        inputs_embeds.requires_grad_(True)
        #inputs_embeds.to(torch.device("cpu"))
        # Identify the position(s) of the word 'group' in the sentence
        group_token_id = LM.tokenizer.encode(" " + group)
            #group_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
        group_start_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
        group_positions = [group_start_positions + i for i in range(len(group_token_id))]


        mu_prime_g = mu_prime_g.view(1, self.num_tokens, LM.model.config.n_embd).to(device)
        #pdb.set_trace()

        # Replace the embeddings for 'group' with mu_prime_g
        p0=group_positions[0]
        # #print (group_positions,group, "group_positions" , tokenizer.encode(" " + group)[0],tokenizer.encode(" " + group), (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True),inputs['input_ids']  )
        # for pos in group_positions:
        #     inputs_embeds[0, pos, :] = mu_prime_g[0,pos-p0,:]

        mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
        offsets = [pos - p0 for pos in group_positions]

        # Create a tensor of the specific embeddings from mu_prime_g to use for replacement
        # Assuming offsets is a list of integers

        replacement_embeddings = torch.stack([mu_prime_g[0, offset, :] for offset in offsets])

        # # Now, iterate over group_positions and replacement_embeddings to update inputs_embeds
        # for i, pos in enumerate(group_positions):
        #     inputs_embeds[0, pos, :] = replacement_embeddings[i]


        new_inputs_embeds = inputs_embeds.clone()  # Create a clone to preserve the original data

        # Now, iterate over group_positions and replacement_embeddings to update new_inputs_embeds
        for i, pos in enumerate(group_positions):
            new_inputs_embeds[0, pos, :] = replacement_embeddings[i]


        # Ensure new_inputs_embeds requires gradients
        new_inputs_embeds.requires_grad_(True)

        #print ("new inputs_embeds.requires_grad", new_inputs_embeds.requires_grad, new_inputs_embeds.grad)
        # Tokenize the trait to get its token IDs

        # Extend inputs_embeds to accommodate trait prediction if necessary
        # required_length = inputs_embeds.size(1) + len(trait_ids) -1   # Current length + trait tokens - 1
        # if inputs_embeds.size(1) < required_length:
        #     extension = torch.zeros((1, len (trait_ids), inputs_embeds.size(2)), device=device)
        #     new_inputs_embeds = torch.cat([new_inputs_embeds, extension], dim=1)
        required_length = inputs_embeds.size(1) + 5 -1   # Current length + trait tokens - 1
        if inputs_embeds.size(1) < required_length:
            extension = torch.zeros((1, 5, inputs_embeds.size(2)), device=device)
            new_inputs_embeds = torch.cat([new_inputs_embeds, extension], dim=1)

        # Calculate logits with extended inputs_embeds
        outputs = LM.model(inputs_embeds=new_inputs_embeds, return_dict=True)
        #outputs= self.to_cpu(outputs)

        logits = outputs['logits']
        logits.to(self.device)
        #outputs = LM.model(inputs_embeds=inputs_embeds)
        #new_inputs_embeds=new_inputs_embeds.detach()
        #extension=extension.detach()

        #logits=logits.detach()
        #replacement_embeddings=replacement_embeddings.detach()
        #mu_prime_g=mu_prime_g.detach()
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = outputs
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = new_inputs_embeds
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = extension
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = replacement_embeddings
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = mu_prime_g
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = logits
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = inputs_embeds
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = mask
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = inputs
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = offsets


        return logits


    def get_log_prob(self, LM, trait,logits, inputs):
      # tokenizer = LM.tokenizer
      # device = self.device
      # gc.collect()


      # # Tokenize the sentence template with the group placeholder
      # # sentence_with_group = sentence_template.format(group=group)
      # # inputs = tokenizer(sentence_with_group, return_tensors='pt')

      # with torch.no_grad():
      #   inputs_embeds = LM.model.transformer.wte(inputs['input_ids'])

      # inputs_embeds.requires_grad_(True)

      # # Identify the position(s) of the word 'group' in the sentence
      # group_token_id = tokenizer.encode(" " + group)
      #   #group_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
      # group_start_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
      # group_positions = [group_start_positions + i for i in range(len(group_token_id))]


      # embedding_size = LM.model.config.n_embd
      # mu_prime_g = mu_prime_g.view(1, self.num_tokens, embedding_size).to(self.device)
      # #pdb.set_trace()

      # # Replace the embeddings for 'group' with mu_prime_g
      # p0=group_positions[0]
      # # #print (group_positions,group, "group_positions" , tokenizer.encode(" " + group)[0],tokenizer.encode(" " + group), (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True),inputs['input_ids']  )
      # # for pos in group_positions:
      # #     inputs_embeds[0, pos, :] = mu_prime_g[0,pos-p0,:]

      # mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
      # offsets = [pos - p0 for pos in group_positions]

      # # Create a tensor of the specific embeddings from mu_prime_g to use for replacement
      # replacement_embeddings = torch.stack([mu_prime_g[0, offset, :] for offset in offsets])

      # # # Now, iterate over group_positions and replacement_embeddings to update inputs_embeds
      # # for i, pos in enumerate(group_positions):
      # #     inputs_embeds[0, pos, :] = replacement_embeddings[i]


      # new_inputs_embeds = inputs_embeds.clone()  # Create a clone to preserve the original data

      # # Now, iterate over group_positions and replacement_embeddings to update new_inputs_embeds
      # for i, pos in enumerate(group_positions):
      #     new_inputs_embeds[0, pos, :] = replacement_embeddings[i]


      # # Ensure new_inputs_embeds requires gradients
      # new_inputs_embeds.requires_grad_(True)

      # #print ("new inputs_embeds.requires_grad", new_inputs_embeds.requires_grad, new_inputs_embeds.grad)
      # # Tokenize the trait to get its token IDs

      # # Extend inputs_embeds to accommodate trait prediction if necessary
      # required_length = inputs_embeds.size(1) + len(trait_ids) -1   # Current length + trait tokens - 1
      # if inputs_embeds.size(1) < required_length:
      #     extension = torch.zeros((1, len (trait_ids), inputs_embeds.size(2)), device=device)
      #     new_inputs_embeds = torch.cat([new_inputs_embeds, extension], dim=1)

      # # Calculate logits with extended inputs_embeds
      # outputs = LM.model(inputs_embeds=new_inputs_embeds, return_dict=True)
      # logits = outputs['logits']
      # #outputs = LM.model(inputs_embeds=inputs_embeds)
      self.count += 1
      if self.count % 500 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        self.count = 0
      # Get the token IDs for the trait
      trait_ids = list(LM.tokenizer.encode(" "+ trait))

      # Directly compute log probabilities in a way that maintains the computational graph
      log_probs = []  # Use a list to collect log probabilities for each trait_id
      start_pos = inputs['input_ids'].size(1)  # Start position for the trait prediction

      for i, trait_id in enumerate(trait_ids):
          token_logits = logits[0, start_pos + i , :]  # Adjusted for 0-indexing
          #print ("token_logits.grad_fn",token_logits.grad_fn)
          #
          #token_prob = torch.softmax(token_logits, dim=-1)[trait_id]
          #token_prob = token_logits[trait_id]
          stable_log_prob = torch.log_softmax(token_logits, dim=-1)[trait_id]

          #log_prob = torch.log(token_prob)
          #print (log_prob.requires_grad, "log_prob")
          #log_probs.append(log_prob)
          log_probs.append(stable_log_prob)

      # Sum the log probabilities to get a single scalar for backpropagation
      total_log_prob = torch.sum(torch.stack(log_probs))
      #print (total_log_prob.device, "total_log_prob" , total_log_prob )
      #print ("total_log_prob.grad_fn",total_log_prob.grad_fn)

      #print (total_log_prob, "total_log_prob")
      #total_log_prob.backward(retain_graph=True
      #token_logits=token_logits.detach()
      #stable_log_prob=stable_log_prob.detach()

      #total_log_prob=total_log_prob.detach()
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = token_logits
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = log_probs
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = total_log_prob
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = stable_log_prob



      return total_log_prob


    def to_cpu(self,obj):
        """
        Recursively move tensors in nested lists, tuples, or dictionaries to CPU.
        """
        if torch.is_tensor(obj):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: self.to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.to_cpu(v) for v in obj)
        else:
            return obj

    def get_top_token_probabilities(self, LM, mu_prime_g, sentence, group, num_predictions=200):
      # Set device for computation
        selected_gpu = sys.argv[1]

        device = torch.device(f'cuda:{selected_gpu}')


        self.top_count += 1
        if self.top_count % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            self.top_count = 0


        # Tokenize the sentence and get the input embeddings
        inputs = LM.tokenizer(sentence, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Get the original embeddings for the entire sentence

        with torch.no_grad():
            inputs_embeds = LM.model.transformer.wte(inputs['input_ids'])

        # Identify the position(s) of the word 'group' in the sentence
        group_token_id = LM.tokenizer.encode(" " + group)
            #group_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]
        group_start_positions = (inputs['input_ids'] == group_token_id[0]).nonzero(as_tuple=True)[1]

        group_positions = [group_start_positions + i for i in range(len(group_token_id))]

        embedding_size = LM.model.config.n_embd  # Get the embedding size from the model configuration

        # Reshape or expand mu_prime_g to match the embedding size and replace the embeddings for 'group'
        mu_prime_g = mu_prime_g.view(1, -1, embedding_size).to(device)  # Adjusted for dynamic size
        for pos in group_positions:
            inputs_embeds[0, pos, :] = mu_prime_g[:, pos - group_positions[0], :]

        # Calculate the output with the modified embeddings
        with torch.no_grad():
            outputs = LM.model(inputs_embeds=inputs_embeds)
        #outputs= self.to_cpu(outputs)
        logits= outputs['logits']
        #print ("outputs",outputs)
        #   if isinstance(outputs, tuple):
        # # Outputs are returned as a tuple, last hidden states are the first element
        #       last_hidden_states = outputs[0]
        #   else:
        #       # Outputs are returned as an object, access last hidden states via .hidden_states
        #       last_hidden_states = outputs.hidden_states

        #print ("last_hidden states",last_hidden_states)

        # Pass the last hidden states through the lm_head to get logits
        #logits = LM.model.lm_head(last_hidden_states[0])
        #logits = outputs.logits

        # Get the logits for the last token
        last_token_logits = logits[0, -1, :]
        last_token_logits = last_token_logits.to(self.device)
        #print ("last_token_logits", last_token_logits.requires_grad)
        #pdb.set_trace()
        #print ("last_token_logits",last_token_logits)
        # Calculate probabilities and get top predictions
        #print ("last_token_logits", last_token_logits)

        #top_probs, top_indices = torch.softmax(last_token_logits, dim=-1).topk(num_predictions)
        top_logits, top_indices = last_token_logits.topk(num_predictions)



        # if self.num_epochs >1:
        #     pdb.set_trace()
        
        
        top_token_probs_dict = {}

        for idx, prob in zip(top_indices, top_logits):
            try:
                # Attempt to decode the index
                token = LM.tokenizer.decode([idx.item()])
                top_token_probs_dict[token] = prob
            except Exception as e:
                # If decoding fails, print the problematic index and the error
                print(f"Failed to decode index {idx.item()}: {e}")
                # Optionally, you can continue to the next iteration without adding to the dict
                continue

        # Convert top predictions to tokens. Avoid detaching or converting to items to maintain gradients
        #top_token_probs_dict = {LM.tokenizer.decode([idx.item()]): prob for idx, prob in zip(top_indices, top_logits)}
        # del top_indices, top_logits
        # del inputs_embeds, inputs
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = logits
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = outputs
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = last_token_logits
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = inputs_embeds
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = inputs
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = mu_prime_g



        return top_token_probs_dict, logits



    # def compute_weighted_loss(self, terms):
    #     # Determine the adjustment constant based on the minimum value of terms
    #     adjustment_constant = torch.max(terms) if torch.max(terms) > 0.02 else 0.02
    #     # Adjust the terms by adding the adjustment constant
    #     adjusted_terms = -terms + adjustment_constant
    #     # Apply softplus function to the negative of the adjusted terms
    #     weight_factor = F.softplus(-adjusted_terms)
    #     # Multiply the weight factor by the original terms
    #     weighted_terms = weight_factor * terms
    #     print("compute_weighted_loss omid")
    #     return weighted_terms

    
    def NoiseEval(self, LM, mu_prime, sampling_group_pairs, traits, PLM=None, tmplt="The <group> is <mask>"):
        batch_size = mu_prime.size(0)

        # Assuming dimensions for 'sampling_group_pairs' and 'traits' are known or can be calculated
        num_groups = len(sampling_group_pairs) * 2  # Assuming each pair has 2 groups
        num_traits = sum(len(ts) for ts in traits.values())

        # Pre-allocate tensors for KL divergences, initialized to zeros
        #kl_divs_tensor = torch.zeros(batch_size, device=self.device, requires_grad=True)
        kl_div_averages_list=[]
        list_of_dfs = []
        gc.collect()
        torch.cuda.empty_cache()
        #tmp = tmplt.replace('<group>', "person")
        tmp="They are <mask>"
        # Precompute neutral scores for each trait
        #neutral_scores = {trait: self.get_neutral_score(LM, tmp, trait) for _, ts in traits.items() for trait in ts}

        for b in range(batch_size):
    # Pre-allocate tensors for scores, initialized to zeros
            scores_tensor1 = torch.zeros((len(sampling_group_pairs), num_traits), device=self.device, requires_grad=True)
            scores_tensor2 = torch.zeros((len(sampling_group_pairs), num_traits), device=self.device, requires_grad=True)

                    #neutral_score = neutral_scores[trait]

            for j, group_pair in enumerate(sampling_group_pairs):
                for group_idx, group in enumerate(group_pair):
                    mu_prime_g = mu_prime[b, j, group_idx, :, :].squeeze()

                    input_txt = tmplt.replace('<mask>', '').replace('<group>', group)
                    #print(input_txt)
                    inputs = LM.tokenizer(input_txt, return_tensors='pt')

                    selected_gpu = sys.argv[1]




                    inputs = {key: value.to(torch.device(f'cuda:{selected_gpu}')) for key, value in inputs.items()}
                    logits=self.calculate_logits( LM, tmplt, mu_prime_g, inputs, group)
                    trait_idx = 0  # Index to keep track of trait position across different dimensions

                    for dim, ts in traits.items():
                        for trait in ts:
                            score= self.get_log_prob(LM, trait,logits, inputs)
                            #score = self.get_log_prob(LM, mu_prime_g, inputs, group, trait, PLM)
                            adjusted_score = score+15#torch.exp(score) #- neutral_score.detach()
                            adjusted_score=adjusted_score.to(self.device)
                            #print ("adjusred_score.requires_grad",adjusted_score.requires_grad)
                            #print ("adjusted_score",adjusted_score)
                            # Update scores_tensor1 and scores_tensor2 without breaking gradient flow
                            indices = (torch.tensor([j], device=self.device), torch.tensor([trait_idx], device=self.device))
                            # Example for updating scores_tensor1 without breaking gradient flow
                            if group_idx == 0:
                                # Create a mask where only the specific index to be updated is True
                                mask = torch.zeros_like(scores_tensor1, dtype=torch.bool)
                                mask[j, trait_idx] = True

                                # Use torch.where to selectively update values without breaking gradient flow
                                scores_tensor1 = torch.where(mask, adjusted_score.expand_as(scores_tensor1), scores_tensor1)
                            else:
                                # Similar approach for scores_tensor2
                                mask = torch.zeros_like(scores_tensor2, dtype=torch.bool)
                                mask[j, trait_idx] = True

                                scores_tensor2 = torch.where(mask, adjusted_score.expand_as(scores_tensor2), scores_tensor2)

                            trait_idx += 1  # Move to the next trait

            # Concatenate first, then apply softmax
            concatenated_p1 = torch.cat([scores_tensor1[j] for j in range(len(sampling_group_pairs))])
            concatenated_p2 = torch.cat([scores_tensor2[j] for j in range(len(sampling_group_pairs))])

            # Now apply softmax
            #concatenated_p1_softmax = F.softmax(concatenated_p1, dim=0)
            concatenated_p2_softmax = F.softmax(concatenated_p2, dim=0)


            concatenated_p1_log_softmax = F.log_softmax(concatenated_p1, dim=0)


            # Compute KL divergence on the softmaxed distributions
            kl_div = F.kl_div(concatenated_p1_log_softmax, concatenated_p2_softmax, reduction='mean')


            num_elements = len(sampling_group_pairs)  # Or any other appropriate count that reflects your data structure
            kl_div_average = kl_div / num_elements
            #print ("kl_div.grad", kl_div.requires_grad)

            kl_div_averages_list.append(kl_div_average)
            #df_data = {f'Group {i+1}': scores_tensor1[i].detach().cpu().numpy() for i in range(scores_tensor1.size(0))}
            #df = pd.DataFrame(data=df_data)
            #list_of_dfs.append(df)

# Step 4: Stack the list into a tensor after the loop
        self.kl_divs_tensor = torch.stack(kl_div_averages_list)
        self.kl_divs_tensor.to(self.device)
        #self.kl_divs_tensor.retain_grad()
        #del kl_div_averages_list
        # scores_tensor1=scores_tensor1.detach()
        # scores_tensor2=scores_tensor2.detach()
        # concatenated_p1=concatenated_p1.detach()
        # concatenated_p2=concatenated_p2.detach()
        # concatenated_p1_log_softmax=concatenated_p1_log_softmax.detach()
        # concatenated_p2_softmax= concatenated_p2_softmax.detach()
        # mask=mask.detach()
        # adjusted_score=adjusted_score.detach()
        # mu_prime_g=mu_prime_g.detach()
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = scores_tensor1
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = scores_tensor2
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = concatenated_p1
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = concatenated_p2
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = concatenated_p1_log_softmax
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = concatenated_p2_softmax
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = mask
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = adjusted_score
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = mu_prime_g
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = kl_div_averages_list
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = kl_div_average
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = indices
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = logits
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = inputs

        return -self.kl_divs_tensor
        #return self.compute_weighted_loss(-self.kl_divs_tensor)  
    def merge_and_update_token_probabilities(self, LM, logits_1, logits_2, top_token_probs_1, top_token_probs_2):

      # Convert lists of tuples to dictionaries for easy lookup
      token_probs_dict_1 = dict(top_token_probs_1)
      token_probs_dict_2 = dict(top_token_probs_2)

      # Convert top tokens to sets for easier manipulation
      set_top_tokens_1 = set(token_probs_dict_1.keys())
      set_top_tokens_2 = set(token_probs_dict_2.keys())

      # Find complementary tokens
      complementary_tokens_1 = set_top_tokens_2 - set_top_tokens_1
      complementary_tokens_2 = set_top_tokens_1 - set_top_tokens_2

      # Get the last token logits from each model
      last_token_logits_1 = logits_1[0, -1, :]
      last_token_logits_2 = logits_2[0, -1, :]
      #print ("last_token_logits_2", last_token_logits_2.requires_grad)


      # # Calculate probabilities for complementary tokens
      # token_probs_1 = torch.softmax(last_token_logits_1, dim=-1)
      # token_probs_2 = torch.softmax(last_token_logits_2, dim=-1)

      token_probs_1 = last_token_logits_1
      token_probs_2 = last_token_logits_2

      #pdb.set_trace()
      # Retrieve and update probabilities for complementary tokens
      for token in complementary_tokens_1:
          token_id = LM.tokenizer.encode(token)[0]  # Ensure no special tokens are added
          token_probs_dict_1[token] = token_probs_1[token_id]  # Removed .item() to keep tensor

      for token in complementary_tokens_2:
          token_id = LM.tokenizer.encode(token)[0]  # Ensure no special tokens are added
          token_probs_dict_2[token] = token_probs_2[token_id]  # Removed .item() to keep tensor

      # last_token_logits_1=last_token_logits_1.detach()
      # last_token_logits_2=last_token_logits_2.detach()
      # token_probs_1=token_probs_1.detach()
      # token_probs_2=token_probs_2.detach()


      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = token_probs_dict_1
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = token_probs_dict_2
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = last_token_logits_1
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = last_token_logits_2
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = complementary_tokens_1
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = complementary_tokens_2

      del token_probs_1
      del token_probs_2
      del last_token_logits_1
      del last_token_logits_2
      #pdb.set_trace()
      del complementary_tokens_1
      del complementary_tokens_2
      del set_top_tokens_1
      del set_top_tokens_2
      return token_probs_dict_1, token_probs_dict_2

    #group_pairs= [ ('man', 'woman'),('stepfather', 'stepmother') ,('he', 'she')]

    def kl_perf( self, LM, PLM, mu_prime, prime, group_pairs,  tmplt="The <group> is powerful", num_predictions=200):

        # Initialize lists to store probabilities for concatenation
        # concatenated_probs_lm_group1 = []
        # concatenated_probs_plm_group1 = []
        # concatenated_probs_lm_group2 = []
        # concatenated_probs_plm_group2 = []
        # batch_kl_divergence_group1=[]
        #batch_kl_divergence_group2=[]
        #pdb.set_trace()
        gc.collect()
        torch.cuda.empty_cache()
        kl_divergence_group1=torch.zeros(mu_prime.size(0) , len (group_pairs), requires_grad= True, device= self.device)
        kl_divergence_group2=torch.zeros(mu_prime.size(0) , len (group_pairs), requires_grad= True, device= self.device)
        concatenated_probs_lm_group1 = torch.empty((0,), device=self.device)
        concatenated_probs_plm_group1 = torch.empty((0,), device=self.device)
        concatenated_probs_lm_group2 = torch.empty((0,), device=self.device)
        concatenated_probs_plm_group2 = torch.empty((0,), device=self.device)

        for batchid in range(mu_prime.size(0)):
            for group_pair_id, (group1, group2) in enumerate(group_pairs):
                # Retrieve embeddings for each group in the pair for LM and PLM
                mu_prime_g1_lm = mu_prime[batchid,group_pair_id, 0,:, :].squeeze(0)
                mu_prime_g2_lm = mu_prime[batchid,group_pair_id, 1,:, :].squeeze(0)
                mu_prime_g1_plm = prime[batchid,group_pair_id, 0, :,:].squeeze(0)
                mu_prime_g2_plm = prime[batchid,group_pair_id, 1, :,:].squeeze(0)

                # Calculate top token probabilities and logits using LM and PLM for each group
                top_token_probs_1_lm, logits_1_lm = self.get_top_token_probabilities(LM, mu_prime_g1_lm, tmplt.replace("<group>", group1), group1, num_predictions)
                top_token_probs_1_plm, logits_1_plm = self.get_top_token_probabilities(PLM, mu_prime_g1_plm, tmplt.replace("<group>", group1), group1, num_predictions)
                top_token_probs_2_lm, logits_2_lm = self.get_top_token_probabilities(LM, mu_prime_g2_lm, tmplt.replace("<group>", group2), group2, num_predictions)
                top_token_probs_2_plm, logits_2_plm = self.get_top_token_probabilities(PLM, mu_prime_g2_plm, tmplt.replace("<group>", group2), group2, num_predictions)

                # Merge and update token probabilities for LM and PLM for each group
                merged_probs_1_lm, merged_probs_1_plm = self.merge_and_update_token_probabilities(LM, logits_1_lm, logits_1_plm, top_token_probs_1_lm, top_token_probs_1_plm)
                merged_probs_2_lm, merged_probs_2_plm = self.merge_and_update_token_probabilities(LM, logits_2_lm, logits_2_plm, top_token_probs_2_lm, top_token_probs_2_plm)

                # Example modification to handle dictionary values as tensors
                concatenated_probs_lm_group1 = [torch.tensor(v, device=self.device, dtype=torch.float, requires_grad=False) for v in merged_probs_1_lm.values()]
                concatenated_probs_plm_group1 = [torch.tensor(v, device=self.device, dtype=torch.float, requires_grad=False) for v in merged_probs_1_plm.values()]
                concatenated_probs_lm_group2 = [torch.tensor(v, device=self.device, dtype=torch.float, requires_grad=False) for v in merged_probs_2_lm.values()]
                concatenated_probs_plm_group2 = [torch.tensor(v, device=self.device, dtype=torch.float, requires_grad=False) for v in merged_probs_2_plm.values()]

                # Convert lists of tensors to a single tensor while preserving the computational graph
                concatenated_probs_lm_group1_tensor = torch.stack(concatenated_probs_lm_group1)
                concatenated_probs_plm_group1_tensor = torch.stack(concatenated_probs_plm_group1)
                concatenated_probs_lm_group2_tensor = torch.stack(concatenated_probs_lm_group2)
                concatenated_probs_plm_group2_tensor = torch.stack(concatenated_probs_plm_group2)

                # Assuming logits are your raw model outputs
                log_probs_lm_group1 = F.log_softmax(concatenated_probs_lm_group1_tensor, dim=-1)
                probs_plm_group1 = F.softmax(concatenated_probs_plm_group1_tensor, dim=-1)
                #pdb.set_trace()

                val1 =  F.kl_div(
                    log_probs_lm_group1,
                    probs_plm_group1,
                    reduction='mean'
                ).unsqueeze(0)



                log_probs_lm_group2 = F.log_softmax(concatenated_probs_lm_group2_tensor, dim=-1)
                probs_plm_group2 = F.softmax(concatenated_probs_plm_group2_tensor, dim=-1)

                val2 = F.kl_div(
                    log_probs_lm_group2,
                    probs_plm_group2,
                    reduction='mean'
                ).unsqueeze(0)
                mask = torch.zeros_like(kl_divergence_group1, dtype=torch.bool)
                mask[batchid, group_pair_id] = True

                # Use torch.where to selectively update values without breaking gradient flow
                kl_divergence_group1 = torch.where(mask, val1.expand_as(kl_divergence_group1), kl_divergence_group1)
                kl_divergence_group2 = torch.where(mask, val2.expand_as(kl_divergence_group2), kl_divergence_group2)
                #print ("concatenated_probs_plm_group2.requires_grad",concatenated_probs_plm_group2_tensor.requires_grad)

        klG1 = kl_divergence_group1.mean(dim=1)  # Mean across group pairs
        klG2 = kl_divergence_group2.mean(dim=1)
        #print ("kl_divergence_group_updated.shape",kl_divergence_group1.shape, klG1.shape ,kl_divergence_group2.shape,klG2.shape )
        #print ("KLG1",klG1.requires_grad)
        # concatenated_probs_lm_group1_tensor=concatenated_probs_lm_group1_tensor.detach()
        # concatenated_probs_plm_group1_tensor=concatenated_probs_plm_group1_tensor.detach()
        # concatenated_probs_lm_group2_tensor=concatenated_probs_lm_group2_tensor.detach()
        # concatenated_probs_plm_group2_tensor=concatenated_probs_plm_group2_tensor.detach()
        # kl_divergence_group1=kl_divergence_group1.detach()
        # kl_divergence_group2=kl_divergence_group2.detach()
        # log_probs_lm_group1=log_probs_lm_group1.detach()
        # probs_plm_group1=probs_plm_group1.detach()
        # del concatenated_probs_lm_group1
        # del concatenated_probs_plm_group1
        # del concatenated_probs_lm_group2
        # del concatenated_probs_plm_group2
        # del kl_divergence_group1
        # del kl_divergence_group2
        # del concatenated_probs_lm_group1_tensor
        # del concatenated_probs_plm_group1_tensor
        # del concatenated_probs_lm_group2_tensor
        # del concatenated_probs_plm_group2_tensor
        # del log_probs_lm_group1
        # del probs_plm_group1
        # del log_probs_lm_group2
        # del probs_plm_group2
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = concatenated_probs_lm_group1_tensor
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = concatenated_probs_lm_group2_tensor
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = concatenated_probs_plm_group1_tensor
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = concatenated_probs_plm_group2_tensor
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = kl_divergence_group1
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = kl_divergence_group2
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = log_probs_lm_group1
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = probs_plm_group1
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = log_probs_lm_group2
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = probs_plm_group2


        return klG1, klG2


class LearningAgent(object):

    def __init__(self, batch_size,PLM, actor, noise_obj, lambda1, traits, sampling_group_pairs, device,gamma=0.5, clip_param=0.2, num_epochs=2, actor_lr=5e-4, learning_template= "The {} is " ):
        self.chkpt_path="/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/checkpoint2.pth"
        self.device = device#T.device('cuda')
        self.lambda1  = lambda1
        #self.tau = tau
        self.batch_size = batch_size
        self.reward= None
        #self.actor = actor # ActorNetwork( 'Actor', PLM)
        self.noise = noise_obj
        self.traits= traits
        self.num_traits = len(traits)  # Number of Traits
        self.gamma=0.5
        # self.groups_sing =self.groups["groups_sing"]# ['man', 'woman']
        # self.groups_plur=self.groups["groups_plur"]#['men', 'women']
        # self.groups_cap_plur=self.groups["groups_cap_plur"] #['Men', 'Women']
        #self.base_group_variations= ['man', 'men', 'Men']
        # self.prior_g_sing='person'
        # self.prior_g_plur='people'
        # self.prior_g_cap_plur='People'
        #self.prime= self.choose_action (PLM,False)
        #self.prime, _= self.noise.sample_batch(PLM, PLM, False)
        self.old_actor=PLM
        self.clip_param=clip_param
        self.num_epochs=num_epochs
        self.actor_lr=actor_lr
        #self.gamma=0.95
        self.embedding_size = actor.model.config.n_embd
        self.sampling_group_pairs=sampling_group_pairs# [('man', 'woman') , ('stepfather', 'stepmother')]
        self.pairs_count= len (self.sampling_group_pairs)
        self.pair_size= len (self.sampling_group_pairs[0])
        self.trait_count= len(traits.keys())
        self.template= "The {group} is {trait}"#learning_template #"The {} is "
        self.Policy_Loss = []
        self.start_save=None
        self.garbage_dict_count =0
        self.garbage_dict = {}
        self.last_id=None



    def calculate_action_log_probs(self,  mu_prime_Batch): # in edit it requires multiple, exclude softmax, dimensions, multi trait , change self.actor to actor etc....
      action_log_probs_batch = torch.zeros(self.batch_size, self.pairs_count, self.pair_size, self.trait_count, device=self.device)
      tokenizer = self.actor.tokenizer

      for e in range(self.batch_size):
          for pair_id, pairs in enumerate(self.sampling_group_pairs):
              for group_id, group_word in enumerate(pairs):
                  # Form the sentence for group_word, will update with trait later
                  sentence_template = self.template.format(group=group_word, trait="{trait}")
                  group_word_tokens = tokenizer.tokenize(" " +group_word)
                  group_word_ids = tokenizer.convert_tokens_to_ids(group_word_tokens)
                  num_group_tokens = len(group_word_ids)

                  # Embed the group_word
                  mu_prime_g = mu_prime_Batch[e, pair_id, group_id, :num_group_tokens, :].view(1, num_group_tokens, -1).to(self.device)

                  for tid, ts in enumerate(self.traits.values()):
                      for subtid, trait in enumerate(ts):
                          # Complete the sentence with the trait
                          complete_sentence = sentence_template.format(trait=trait)
                          inputs = tokenizer(complete_sentence, return_tensors='pt')
                          inputs = {key: value.to(self.device) for key, value in inputs.items()}

                          # Get input embeddings
                          inputs_embeds = self.actor.model.transformer.wte(inputs['input_ids'])
                          # Replace embeddings for the group_word part with mu_prime_g
                          inputs_embeds[:, 1:1 + num_group_tokens] = mu_prime_g

                          with torch.no_grad():
                              # Get logits for the entire sequence
                              group_outputs = self.actor.model(inputs_embeds=inputs_embeds)
                              group_logits = group_outputs.logits

                          # Calculate probabilities for the trait
                          trait_tokens = tokenizer.tokenize(" "+trait)
                          trait_ids = tokenizer.convert_tokens_to_ids(trait_tokens)
                          seq_prob = torch.tensor(1.0, device=self.device,requires_grad=True)

                          # Iterate through each token in the trait to calculate its probability
                          for idx, t_idx in enumerate(trait_ids):
                              # Assuming the trait is at the end, find its starting position
                              trait_start_pos = inputs['input_ids'].size(1) - len(trait_ids)
                              token_logits = group_logits[:, trait_start_pos + idx, :]
                              token_prob = torch.softmax(token_logits, dim=-1)[0, t_idx]
                              seq_prob = seq_prob* token_prob

                          # Convert sequence probability to log probability
                          log_prob = torch.log(seq_prob.clamp(min=1e-10))
                          action_log_probs_batch[e, pair_id, group_id, tid] = log_prob

      return action_log_probs_batch


    # def calculate_action_log_probs(self, actor, mu_prime_Batch):
    #     action_log_probs_batch = torch.zeros(self.batch_size, self.pairs_count, self.pair_size, self.trait_count)
    #     tokenizer = actor.tokenizer

    #     for e in range(self.batch_size):
    #         for pair_id, pairs in enumerate(self.sampling_group_pairs):
    #             for group_id, group_word in enumerate(pairs):A
    #                 sentences = [self.template.format(group_word) for _ in traits.values()]
    #                 inputs = tokenizer(sentences, return_tensors='pt')
    #                 inputs = {key: value.to(self.device) for key, value in inputs.items()}

    #                 with torch.no_grad():
    #                     inputs_embeds = actor.model.transformer.wte(inputs['input_ids'])

    #                 # Tokenize group_word and count the tokens (k)
    #                 tokenized_group_word = tokenizer.tokenize(group_word)
    #                 k = len(tokenized_group_word)  # Number of tokens

    #                 # Ensure k does not exceed num_tokens
    #                 k = min(k, self.num_tokens)

    #                 # Adjust mu_prime_g to read k slices
    #                 mu_prime_g = mu_prime_Batch[e, pair_id, group_id, 0:k, :].view(1, k, self.embedding_size).to(self.device)

    #                 # Update inputs_embeds with sliced mu_prime_g
    #                 inputs_embeds[:, 1:1 + k, :] = mu_prime_g

    #                 with torch.no_grad():
    #                     group_outputs = actor.model(inputs_embeds=inputs_embeds)
    #                     group_logits = group_outputs.logits
    #                     for tid, ts in enumerate(traits.values()):
    #                       for subtid, trait in enumerate(ts):
    #                           trait_with_prefix = " " + trait
    #                           trait_tokens = tokenizer.tokenize(trait_with_prefix)
    #                           trait_ids = tokenizer.convert_tokens_to_ids(trait_tokens)

    #                           # Initialize variables for probabilities
    #                           seq_prob = 1.0
    #                           probs = []
    #                           log_probs_with_epsilon=[]
    #                           # Ensure there's enough space in inputs_embeds to add the trait tokens
    #                           # Adjust this part according to your specific model and sentence structure
    #                           inputs_embeds = adjust_inputs_embeds_for_trait(inputs_embeds, len(trait_ids))

    #                           for idx, t_idx in enumerate(trait_ids):
    #                               # Generate the embedding for the current trait token
    #                               token_embed = actor.model.transformer.wte(torch.tensor([[t_idx]], device=self.device))

    #                               # Determine the position to place the token embedding
    #                               token_position = -len(trait_ids) + idx
    #                               modified_inputs_embeds = inputs_embeds.clone()
    #                               modified_inputs_embeds[:, token_position, :] = token_embed.squeeze(0)

    #                               with torch.no_grad():
    #                                   # Get logits for the modified sequence
    #                                   modified_group_outputs = actor.model(inputs_embeds=modified_inputs_embeds)
    #                                   modified_group_logits = modified_group_outputs.logits

    #                               # Get the logits for the current trait token position
    #                               token_logits = modified_group_logits[:, token_position, :]
    #                               token_prob = torch.softmax(token_logits, dim=-1)[:, t_idx]
    #                               seq_prob *= token_prob

    #                           # Store the final sequence probability
    #                           probs.append(seq_prob)
    #                           epsilon = 1e-10

    #                           for p in probs:
    #                               log_probs_with_epsilon.append(torch.clamp(p, min=epsilon))

    #                           log_values = torch.log(torch.stack(log_probs_with_epsilon, dim=-1))
    #                           log_prob = torch.mean(log_values, dim=-1)
    #                           print ("log_prob", log_prob)
    #                           pdb.set_trace()
    #                           print ("action_log_probs_batch[e, pair_id, group_id, tid + (self.trait_count * subtid)]",action_log_probs_batch[e, pair_id, group_id, tid + (self.trait_count * subtid)])
    #                           action_log_probs_batch[e, pair_id, group_id, tid + (self.trait_count * subtid)] = log_prob

    #     return action_log_probs_batch




    def calculate_advantage_zero_noise(self, noise_obj, reward_obj):

        return 0

    # def sample_terms(self, noise_obj, reward_obj, PLM ):
    #     mu_prime_Batch, _ = noise_obj.sample_batch(self.actor, PLM, True)
    #     #mu_prime_Batch = noise_obj.sample_batch(actor, PLM, False)

    #     batch_rewards = reward_obj.calculate_reward(mu_prime_Batch, self.actor, PLM, self.lambda1)
    #     #batch_rewards = torch.tensor(batch_rewards, dtype=torch.float64, device=self.device, requires_grad=True)
    #     #batch_rewards = batch_rewards.clone().detach().requires_grad_(True)
    #     batch_rewards = batch_rewards.clone().requires_grad_(True)

    #     batch_rewards = batch_rewards.to(self.device).double()  # Ensure it's on the correct device and in double precision


    #     #print ("mu_prime_Batch size", mu_prime_Batch.size())
    #     self.num_tokens=noise_obj.num_tokens
    #     #action_log_probs_batch=self.calculate_action_log_probs(self.actor, mu_prime_Batch)
    #     #old_action_log_probs_batch=self.calculate_action_log_probs(self.old_actor, mu_prime_Batch)
    #     #self.old_actor = self.actor
    #     R0=self.calculate_advantage_zero_noise(noise_obj, reward_obj)

    #     adv_targ= self.calculate_advantage(batch_rewards, R0)
    #     adv_targ.retain_grad()

    #     #add more explainable terms??
    #     #return (R0, adv_targ, action_log_probs_batch, old_action_log_probs_batch)
    #     return (R0, adv_targ)


    def sample_terms(self, noise_obj, reward_obj, PLM, actor ):

        mu_prime_Batch,_ = noise_obj.sample_batch(actor, PLM, True)
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = mu_prime_Batch
        self.garbage_dict_count += 1
        self.garbage_dict[self.garbage_dict_count] = _

        return (mu_prime_Batch )

    def free_memory(self,obj):
        """
        Recursively move tensors in nested lists, tuples, or dictionaries to CPU.
        """

        if torch.is_tensor(obj):
            return obj.detach()
        elif isinstance(obj, dict):
            return {k: self.free_memory(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.free_memory(v) for v in obj)
        else:
            return obj

    def delete_memory(self,obj):
        """
        Recursively move tensors in nested lists, tuples, or dictionaries to CPU.
        """

        if torch.is_tensor(obj):
            obj= None
            del obj
        elif isinstance(obj, dict):
            return {k: self.delete_memory(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.delete_memory(v) for v in obj)
        else:
            return obj
    
    

        
    def calculate_prob(self,LM, inputs):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        selected_gpu = sys.argv[1]
        device = torch.device(f'cuda:{selected_gpu}')

        # Tokenize traits and convert to IDs in a batch
        traits_with_prefix = "calm"
        trait_tokens = tokenizer(traits_with_prefix, return_tensors='pt')
        trait_ids = trait_tokens['input_ids'].to(device)
        
        input_ids = tokenizer.encode(inputs, return_tensors='pt').to(torch.device(f'cuda:{selected_gpu}'))  # Ensure tensors are on the same device as model

        with torch.no_grad():
                outputs = LM(input_ids=input_ids)
                logits = outputs.logits
        print ("logits.size()",logits.size())
            # Iterate over the trait_ids and sum their corresponding logits

        for i, trait_id in enumerate(trait_ids):
            token_logits = logits[0, -1, :]
            trait_logit = token_logits[trait_id]  # Get the logit for the current trait_id
            print ("trait_logit",trait_logit )

        return trait_logit



    def compare_model_parameters(self,model, checkpoint):
        saved_state_dict = checkpoint['model_state_dict']
        parameters_changed = False

        for param_name, param in model.named_parameters():
            print("Checking parameter:", param_name)
            if param.requires_grad:
                print("some params requirs grad")
                # Convert the current parameter to numpy for comparison
                current_param_np = param.detach().cpu().numpy()

                # Check if the current parameter exists in the saved state dict and requires_grad
                if param_name in saved_state_dict:
                    # Extract the corresponding parameter from the saved state dict
                    saved_param = saved_state_dict[param_name].detach().cpu().numpy()
                    #print ( "saved_param",saved_param)
                    # Compare the current parameter with the saved parameter
                    if not np.array_equal(current_param_np, saved_param):
                        print(f"Trainable parameter '{param_name}' has changed.")
                        parameters_changed = True
                    else:
                        print(f"Trainable parameter '{param_name}' remains unchanged.")
                else:
                    print(f"Trainable parameter '{param_name}' not found in saved checkpoint.")
            else:
                print(f"Parameter '{param_name}' is not set to be trainable.")

        if not parameters_changed:
            print("No trainable parameters have changed.")
        else:
            print("Some trainable parameters have changed.")

    def test_saved_model(self):
        

        device = torch.device('cuda')
        LM_name = "gpt-2"
        actor_lr=2e-4
        ###actor settings
        in_net=False
        in_net_init_identity=False
        out_net=False
        out_net_init_identity=False
        freeze_ln=False
        freeze_pos=False
        freeze_wte=True

        freeze_ff=True
        freeze_attn=True
        dup_lm_head=False
        dup_lm_head_bias=False

        PLM= LanguageModel('gpt-2')#.to(device)
        actor =ActorNetwork('actor', PLM, device, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, chkpt_dir)#.to(device)

        drive_folder = "/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/"
        checkpoint_filename = "checkpoint2.pth"  # Name of the checkpoint file
        checkpoint_file = os.path.join(drive_folder, checkpoint_filename)


        #device=torch.device('cpu')
        PLM= LanguageModel(LM_name)#.to(device)
        PLM_model=PLM.model
        #lm_head_model =ActorNetwork('actor', PLM, device, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, chkpt_dir).to(device)
        lm_head_model= PLM.model

        selected_gpu = sys.argv[1]

        
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(f'cuda:{selected_gpu}'))

        model_state = lm_head_model.state_dict()

        compatible_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() }

        # Print model state dict keys



        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            # Adjust the key name as needed to match the model's expected keys
            new_key = key  # Modify this as needed based on your model's architecture
            new_state_dict[new_key] = value

        # Now try loading the adjusted state dictionary
        model_state.update(new_state_dict)
        lm_head_model.load_state_dict(model_state, strict=True)


        print("Compatible keys found:")
        for key in compatible_state_dict.keys():
            print(key)


        model = lm_head_model.to(device)
        print("Model's state dict keys:")
        for key in model.state_dict().keys():
            print(key)

        # Assuming 'checkpoint' is your loaded checkpoint dictionary
        print("\nCheckpoint's state dict keys:")
        for key in checkpoint['model_state_dict'].keys():
            print(key)


        # self.compare_model_parameters(PLM_model, checkpoint)
        # print( compatible_state_dict.keys())

        inputs= ["When I was a schoolboy, I was always ", "When I was a schoolgirl, I was always BLANK" , "He was alway ", "She was always "]
        for inp in inputs:
            print (inp, self.calculate_prob(model, inp) ,self.calculate_prob(PLM_model,inp) )





    def ppo_update(self, samples, noise_obj, reward_obj, PLM,actor, epoch):

        
        (mu_prime_Batch)=samples
        #batch_rewards = reward_obj.calculate_reward(mu_prime_Batch, actor, PLM, self.lambda1,True)
        batch_rewards = reward_obj.calculate_reward(mu_prime_Batch, actor, PLM, self.lambda1)
        batch_rewards = batch_rewards.clone().requires_grad_(True)
        batch_rewards.retain_grad()
        batch_rewards = batch_rewards.to(self.device).double()  # Ensure it's on the correct device and in double precision
        self.num_tokens=noise_obj.num_tokens
        #action_log_probs_batch=self.calculate_action_log_probs(self.actor, mu_prime_Batch)
        #old_action_log_probs_batch=self.calculate_action_log_probs(self.old_actor, mu_prime_Batch)
        #self.old_actor = self.actor
        R0=self.calculate_advantage_zero_noise(noise_obj, reward_obj)
        #gc.set_debug(gc.DEBUG_LEAK)

        next_state_estimate = torch.mean(batch_rewards)  # Keep as tensor for gradient tracking
        next_state_estimate.retain_grad()

        adv_targ=next_state_estimate
        adv_targ.retain_grad()
        policy_loss= -adv_targ *200#.mean()
        print ("policy loss magnified by 100", policy_loss)
        policy_loss.retain_grad()

        #self.actor.optimizer =optim.Adam(self.actor.model.parameters(), self.actor_lr)
        # Define the scheduler with exponential decay
        #self.actor.model.register_backward_hook(backward_hook)
        #self.actor.model.transformer.wte.register_full_backward_hook(full_backward_hook)
        #self.actor.model.transformer.h[0].register_full_backward_hook(full_backward_hook)
        #self.actor.model.register_full_backward_hook(full_backward_hook)
        actor.optimizer.zero_grad()
        actor.model.train()
        #pdb.set_trace()

        self.Policy_Loss.append(policy_loss)
        policy_loss.backward()

        actor.optimizer.step()
        actor.scheduler.step()

        # for obj in reward_obj.garbage_dict.values():
        #   self.free_memory(obj)
        #   self.delete_memory(obj)

        # del reward_obj.garbage_dict


        memory_manager = MemoryManager()
        memory_manager.free_memory(reward_obj.garbage_dict)
        memory_manager.delete_memory(reward_obj.garbage_dict)
        reward_obj.garbage_dict={}

        mu_prime_Batch.detach()
        del mu_prime_Batch
        batch_rewards.detach()
        del batch_rewards
        next_state_estimate.detach()
        del next_state_estimate
        adv_targ.detach()
        del adv_targ
        del R0

        gc.collect()
        torch.cuda.empty_cache()
        inputsentences= ["The schoolgirl is  ", "The schoolboy is  " , "He is ", "She is"]
        for inp in inputsentences:
            
            print (inp, self.calculate_prob(actor.model, inp) ,self.calculate_prob(PLM.model, inp) )

        
        
        # for obj in gc.garbage:
        #   print(obj)

        #print (policy_loss, "policy_loss grad", policy_loss.grad, "batch_rewards", batch_rewards.grad)
        #print("reward_obj.kl_divs_tensor.grad", reward_obj.kl_divs_tensor.grad)
        #print ("reward_obj.last_hidden_states[0] grad ",reward_obj.last_hidden_states[0].grad)
        # for name, parameter in self.actor.model.named_parameters():
        #   if parameter.grad is not None:
        #       print(f"{name} gradient norm: {parameter.grad.norm().item()}")

        #   else:
        #       print(f"{name} has no gradient")
        #       print ("parameter.requires_grad",parameter.requires_grad)
    # def calculate_advantage(self, reward_batch, R0):

    #     next_state_estimate = torch.mean(reward_batch)  # Keep as tensor for gradient tracking
    #     next_state_estimate.retain_grad()

    #     #print ("next_state_estimate.grad", next_state_estimate.grad )
    #     # Compute advantage
    #     # advantage = torch.zeros(self.batch_size, device=reward_batch.device)  # Ensure same device as reward_batch

    #     # for e in range(self.batch_size):  # Ensure correct looping
    #     #     # Compute advantage without converting to scalar
    #     #     # Assuming R0 is a tensor, remove .item() to keep the calculation within the computational graph
    #     #     advantage[e] = reward_batch[e] + self.gamma * next_state_estimate - R0

    #     # If you need to print or log values, you can convert to scalar outside the gradient computation
    #     print("batch reward", [rew.item() for rew in reward_batch])

    #     return next_state_estimate  # Keep as tensor if used in subsequent gradient computations



    # def ppo_update(self, samples):

    #     # ??(share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
    #     #  actions_batch, value_preds_batch, return_batch, masks_batch,
    #     #  active_masks_batch, old_action_log_probs_batch, adv_targ,
    #     #  available_actions_batch) = samples
    #     #(R0, adv_targ, action_log_probs_batch, old_action_log_probs_batch)=samples
    #     (R0, adv_targ)=samples

    #     # Calculate the main PPO-clip objective
    #     #imp_weights = torch.prod(torch.exp(action_log_probs_batch - old_action_log_probs_batch), dim=-1, keepdim=True)
    #     #exp_diff = torch.exp(action_log_probs_batch - old_action_log_probs_batch)
    #     # Multiply across dimensions 1, 2, and 3

    #     #imp_weights = exp_diff.prod(dim=1, keepdim=True)
    #     #imp_weights = imp_weights.prod(dim=2, keepdim=True)
    #     #imp_weights = imp_weights.prod(dim=3, keepdim=True)
    #     #imp_weights_clamped = torch.clamp(imp_weights, min=None, max=1)


    #     #surr1 = imp_weights_clamped * adv_targ
    #     #surr2 = torch.clamp(imp_weights_clamped, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
    #     #policy_loss = -torch.sum(torch.min(surr1, surr2), dim=0, keepdim=True) #- R0
    #     #print ("policy loss", policy_loss , "imp_weights" , imp_weights)

    #     # # Actor update
    #     # print ("actor model grad requires?")
    #     # for param in self.actor.model.parameters():
    #     #     print(param.requires_grad)

    #     policy_loss= -adv_targ #.mean()
    #     print ("policy loss", policy_loss)
    #     policy_loss.retain_grad()

    #     #self.actor.optimizer =optim.Adam(self.actor.model.parameters(), self.actor_lr)
    #     # Define the scheduler with exponential decay
    #     #self.actor.model.register_backward_hook(backward_hook)
    #     #self.actor.model.transformer.wte.register_full_backward_hook(full_backward_hook)
    #     #self.actor.model.transformer.h[0].register_full_backward_hook(full_backward_hook)
    #     #self.actor.model.register_full_backward_hook(full_backward_hook)
    #     self.actor.optimizer.zero_grad()
    #     self.actor.model.train()
    #     #pdb.set_trace()

    #     self.Policy_Loss.append(policy_loss)
    #     policy_loss.backward()

    #     self.actor.optimizer.step()
    #     self.actor.scheduler.step()
    #     print (policy_loss, "policy_loss grad", policy_loss.grad)



    def compare_model_parameters(self, model, checkpoint):
      saved_state_dict = checkpoint['model_state_dict']
      parameters_changed = False

      for param_name, param in model.named_parameters():
          print("Checking parameter:", param_name)
          if param.requires_grad:
              print("some params requirs grad")
              # Convert the current parameter to numpy for comparison
              current_param_np = param.detach().cpu().numpy()

              # Check if the current parameter exists in the saved state dict and requires_grad
              if param_name in saved_state_dict:
                  # Extract the corresponding parameter from the saved state dict
                  saved_param = saved_state_dict[param_name].detach().cpu().numpy()
                  #print ( "saved_param",saved_param)
                  # Compare the current parameter with the saved parameter
                  if not np.array_equal(current_param_np, saved_param):
                      print(f"Trainable parameter '{param_name}' has changed.")
                      parameters_changed = True
                  else:
                      print(f"Trainable parameter '{param_name}' remains unchanged.")
              else:
                  print(f"Trainable parameter '{param_name}' not found in saved checkpoint.")
          else:
              print(f"Parameter '{param_name}' is not set to be trainable.")

      if not parameters_changed:
          print("No trainable parameters have changed.")
      else:
          print("Some trainable parameters have changed.")

      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = saved_state_dict
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = current_param_np
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = checkpoint
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = model
      self.garbage_dict_count += 1
      self.garbage_dict[self.garbage_dict_count] = saved_param




    # def load_checkpoint(self, checkpoint_path):

    #   checkpoint = torch.load(checkpoint_path)
    #   self.actor.load_state_dict(checkpoint['model_state_dict'])
    #   if 'in_net_state_dict' in checkpoint and hasattr(self.actor.model, 'in_net'):
    #       self.actor.model.in_net.load_state_dict(checkpoint['in_net_state_dict'])
    #   if 'out_net_state_dict' in checkpoint and hasattr(self.actor.model, 'out_net'):
    #       self.actor.model.out_net.load_state_dict(checkpoint['out_net_state_dict'])
    #   if 'lm_head_new_state_dict' in checkpoint and hasattr(self.actor.model, 'lm_head_new'):
    #       self.actor.model.lm_head_new.load_state_dict(checkpoint['lm_head_new_state_dict'])
    #   print('Checkpoint loaded successfully from', checkpoint_path)


    def save_models(self,actor):

        # checkpoint = torch.load('/home/oshokrol/zero-shot-2/ActorModelCheckpoints/checkpoint.pth')
        # if checkpoint is not None:
        #     print(checkpoint.keys())

        #     if self.start_save is not None:
        #       self.compare_model_parameters(actor.model, checkpoint)
        #     self.start_save=1
        # else:
        #     print("Checkpoint is None. Check the file path and content.")


        #Compare the current model parameters with those loaded from the checkpoint

        actor.save_checkpoint()

        print ("saved model at this batch")

    import os

    def unique_filename(self, base_filename, extension):
        counter = 1
        filename = f"{base_filename}{extension}"
        while os.path.exists(filename):
            filename = f"{base_filename}_{counter}{extension}"
            counter += 1
        return filename


    def plot_loss_trend(self,reward_obj):

        Policy_Loss = [tensor.item() for tensor in self.Policy_Loss]  # Convert tensors to scalars

        # Plotting
        plt.figure(figsize=(12, 6))  # Adjust the figure size for better visibility
        plt.plot(Policy_Loss, linestyle='-', color='b', alpha=0.7)  # Plot with a blue line
        plt.title('Policy Loss Trend per Iteration')  # Title of the plot
        plt.xlabel('Iteration')  # X-axis label
        plt.ylabel('Loss')  # Y-axis label
        plt.grid(True)  # Show grid

        # Optional: If you want to see a smoother trend, consider plotting a moving average
        window_size = 50  # Define the window size for the moving average
        moving_avg = [sum(Policy_Loss[i:i+window_size])/window_size for i in range(len(Policy_Loss)-window_size+1)]
        plt.plot(range(window_size-1, len(Policy_Loss)), moving_avg, linestyle='-', color='r', label='Moving Average')  # Plot moving average
        unique_plot1_filename = self.unique_filename('plot1', '.png')
        plt.savefig(unique_plot1_filename)
        plt.legend()
        plt.show()


        klp = [tensor.item() for tensor in reward_obj.klp]
       
        plt.figure(figsize=(12, 6))  # Adjust the figure size for better visibility
        plt.plot(klp, linestyle='-', color='b', alpha=0.7)  # Plot with a blue line
        plt.title('klperformance average Trend per Iteration')  # Title of the plot
        plt.xlabel('Iteration')  # X-axis label
        plt.ylabel('klp')  # Y-axis label
        plt.grid(True)  # Show grid

        
        unique_plot2_filename = self.unique_filename('plot2', '.png')
        plt.savefig(unique_plot2_filename)
        plt.legend()
        plt.show()

        # Optional: If you want to see a smoother trend, consider plotting a moving average
        window_size = 50  # Define the window size for the moving average
        moving_avg = [sum(klp[i:i+window_size])/window_size for i in range(len(klp)-window_size+1)]
        plt.plot(range(window_size-1, len(klp)), moving_avg, linestyle='-', color='r', label='Moving Average')  # Plot moving average

        
        NoiseEval = [tensor.item() for tensor in reward_obj.NoiseE]
       
        plt.figure(figsize=(12, 6))  # Adjust the figure size for better visibility
        plt.plot(NoiseEval, linestyle='-', color='b', alpha=0.7)  # Plot with a blue line
        plt.title('noise Eval average Trend per Iteration')  # Title of the plot
        plt.xlabel('Iteration')  # X-axis label
        plt.ylabel('noise')  # Y-axis label
        plt.grid(True)  # Show grid

        # Optional: If you want to see a smoother trend, consider plotting a moving average
        window_size = 50  # Define the window size for the moving average
        moving_avg = [sum(NoiseEval[i:i+window_size])/window_size for i in range(len(NoiseEval)-window_size+1)]
        plt.plot(range(window_size-1, len(NoiseEval)), moving_avg, linestyle='-', color='r', label='Moving Average')  # Plot moving average
        unique_plot3_filename = self.unique_filename('plot3', '.png')
        plt.savefig(unique_plot3_filename)
        plt.legend()
        plt.show()



    # def save_checkpoint(self, actor_model, epoch, base_path='/home/oshokrol/zero-shot-2/ActorModelCheckpoints'):
    #     """
    #     Save the model checkpoint at the end of an epoch.
        
    #     Parameters:
    #     - actor_model: The model to save.
    #     - epoch: The current epoch number.
    #     - base_path: Base directory to save checkpoints.
    #     """
    #     # Ensure the base directory exists
    #     os.makedirs(base_path, exist_ok=True)

    #     # Define a pattern to match filenames and extract IDs
    #     pattern = re.compile(r'checkpoint_epoch{}_exp(\d+).pth'.format(epoch))

    #     # List all files in the base directory
    #     existing_files = os.listdir(base_path)
        
    #     # Filter and sort files by ID for the current epoch
    #     matching_files = sorted(
    #         [f for f in existing_files if pattern.match(f)],
    #         key=lambda x: int(pattern.match(x).group(1))
    #     )

    #     # Determine the next experiment ID
    #     if matching_files:
    #         last_id = int(pattern.match(matching_files[-1]).group(1))
    #         next_id = last_id + 1
    #     else:
    #         next_id = 0

    #     # Define the filename with the next experiment ID
    #     filename = f'checkpoint_epoch{epoch}_exp{next_id}.pth'
    #     filepath = os.path.join(base_path, filename)

    #     # Save the model checkpoint
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': actor_model.state_dict(),
    #     }, filepath)

    #     print(f"Checkpoint saved to {filepath}")


    def save_checkpoint(self, actor_model, epoch, base_path='/home/oshokrol/zero-shot-2/ActorModelCheckpoints1'):
        """
        Save the model checkpoint at the end of an epoch.
        
        Parameters:
        - actor_model: The model to save.
        - epoch: The current epoch number.
        - base_path: Base directory to save checkpoints.
        """
        # Ensure the base directory exists
        os.makedirs(base_path, exist_ok=True)

        # Define a pattern to match filenames and extract IDs
        pattern = re.compile(r'checkpoint_epoch{}_exp(\d+).pth'.format(epoch))

        # For the first epoch, or if last_id has not been set, determine next_id based on existing files
        if epoch == 1 or self.last_id is None:
            # List all files in the base directory
            existing_files = os.listdir(base_path)
            
            # Filter and sort files by ID for the current epoch
            matching_files = sorted(
                [f for f in existing_files if pattern.match(f)],
                key=lambda x: int(pattern.match(x).group(1))
            )

            # Determine the next experiment ID
            if matching_files:
                last_id = int(pattern.match(matching_files[-1]).group(1))
                next_id = last_id + 1
            else:
                next_id = 0
        else:
            # For epochs after the first, increment last_id
            next_id = self.last_id + 1
        
        # Update last_id for subsequent epochs
        self.last_id = next_id

        # Define the filename with the next experiment ID
        filename = f'checkpoint_epoch{epoch}_exp{next_id}.pth'
        filepath = os.path.join(base_path, filename)

        # Save the model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': actor_model.state_dict(),
        }, filepath)

        print(f"Checkpoint saved to {filepath}")


    def learn(self, num_batches, reward_obj, noise_obj, PLM, actor):

        T_max = 100  # Example value, adjust as needed
        # Generate 8 log-linearly spaced learning rates between 1e-4 and 3e-4
        #learning_rates = np.logspace(np.log10(1e-4), np.log10(3e-4), num=1)
        actor.num_epochs= self.num_epochs
        coef= 1
        actor.increment_epoch()

        # selected_gpu = sys.argv[1]
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # device = torch.device(f'cuda:{selected_gpu}')
        # LM_name = "gpt2"
        # actor_lr=8e-5
        # ###actor settings
        # in_net=False
        # in_net_init_identity=False
        # out_net=False
        # out_net_init_identity=False
        # freeze_ln=False
        # freeze_pos=False
        # freeze_wte=False
        # freeze_ff=True
        # freeze_attn=True
        # dup_lm_head=False
        # dup_lm_head_bias=False
        # #chkpt_dir='tmp/RL'
        # ##end of actor settings
                
        



      # later change this for loop to a while loop based on not external_done_signal
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            actor.increment_epoch()
            if epoch<3:
                coef=0.4

            else:
                coef=1

            #self.actor.update_trainable_layers()
            for i in range(0, math.ceil(coef * num_batches)):
                print ("batch:", i)
                print (reward_obj.batch_size,"batchsize")
                samples= self.sample_terms( noise_obj, reward_obj,PLM,actor)
                print("samples finished")
                gc.collect()
                torch.cuda.empty_cache()

                self.ppo_update( samples, noise_obj, reward_obj, PLM,actor, epoch)

                self.save_models(actor)
                reward_obj.num_iterations= i
                print("ppo update finished")
                if num_batches% 10 == 0:                    
                    noise_obj.start_condition= True
                    # del actor
                    # del PLM

                    # PLM= LanguageModel(LM_name).to(device)
                    # checkpoint = torch.load('ActorModelCheckpoints/checkpoint.pth')
                    # actor = ActorNetwork('actor', PLM, device,num_batches,actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias).to(device)
                    # actor.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    # del checkpoint 
                    # gc.collect()                    


            #(noise_object,_,_,_,_)=samples
            #coef= coef*0.63
            print(f"Finished Epoch {epoch + 1}/{self.num_epochs}")
            noise_obj.plot_reward_distribution()
            self.plot_loss_trend(reward_obj)

            reward_obj.num_epochs= self.num_epochs
                        
            print(f"Training completed.for epoch{epoch+1}")

            # Create a deep copy of the actor.model
            model_copy = copy.deepcopy(actor.model)
            self.save_checkpoint(actor.model, epoch + 1)
            overall_results, intrasentence_bias, intersentence_bias = eval_stereoset(model_copy, model_copy, True)
            
            evaluate_pedb( self.device, "0", "/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/checkpoint2.pth", "crows", "gender", 42,  "./pedb_main/results_pedb/")
            evaluate_pedb( self.device, "0", "/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/checkpoint2.pth", "crows", "race", 42,  "./pedb_main/results_pedb/")
            evaluate_pedb( self.device, "0", "/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/checkpoint2.pth", "crows", "religion", 42,  "./pedb_main/results_pedb/")            
            
            del model_copy

            noise_obj.start_condition= True

# def full_backward_hook(module, grad_input, grad_output):
#     print(f"Full backward hook called for {module.__class__.__name__}")
#     for i, grad in enumerate(grad_input):
#         if grad is not None:
#             print(f"Grad input {i} shape: {grad.shape} - requires_grad: {grad.requires_grad}")


def full_backward_hook(module, grad_input, grad_output):

    print(f"Full backward hook called for {module.__class__.__name__}")
    for i, grad in enumerate(grad_input):
        if grad is not None:
            print(f"Grad input {i} shape: {grad.shape}, requires_grad: {grad.requires_grad}")
    for i, grad in enumerate(grad_output):
        if grad is not None:
            print(f"Grad output {i} shape: {grad.shape}, requires_grad: {grad.requires_grad}")

class MemoryManager:
    def free_memory(self, obj):
        """
        Recursively detach tensors in nested data structures.
        """
        if torch.is_tensor(obj):
            return obj.detach()
        elif isinstance(obj, dict):
            return {k: self.free_memory(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.free_memory(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.free_memory(v) for v in obj)
        else:
            return obj

    def delete_memory(self, obj):
        """
        Recursively delete references in nested data structures.
        """
        if torch.is_tensor(obj):
            del obj
        elif isinstance(obj, dict):
            for k in list(obj.keys()):
                self.delete_memory(obj[k])
            obj.clear()
        elif isinstance(obj, list):
            while obj:
                self.delete_memory(obj.pop())
        elif isinstance(obj, tuple):
            # Tuples are immutable, so we cannot clear them or delete their contents in-place.
            # Instead, handle each element if it's a tensor or another mutable structure.
            for item in obj:
                self.delete_memory(item)
        # For other types, there's no direct action to take. Python's garbage collector will handle it.




def main():

    # logging.warning('This is a warning')
    # logging.info('This is an informational message')
    selected_gpu = sys.argv[1]
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    device = torch.device(f'cuda:{selected_gpu}')
    LM_name = "gpt2"
    actor_lr=8e-5
    ###actor settings
    in_net=False
    in_net_init_identity=False
    out_net=False
    out_net_init_identity=False
    freeze_ln=False
    freeze_pos=False
    freeze_wte=False
    freeze_ff=True
    freeze_attn=True
    dup_lm_head=False
    dup_lm_head_bias=False
    #chkpt_dir='tmp/RL'
    ##end of actor settings
    batch_size= 16
    alpha =0.000025
    adv_batch_size=1
    variance_scale_factor=0.2
    template= "the {} is powerful"
    num_samples=30
    maxiter=5
    improvement_threshold=0.0001
    mixed_sampling_weight=0.2
    num_restarts=2
    raw_samples=30
    batch_limit=8
    lambda1=1
    #sampling_group_pairs= [('he', 'she'), ('schoolboy', 'schoolgirl')]
    #sampling_group_pairs= [ ('he', 'she'),('schoolboy', 'schoolgirl'), ('him', 'her'), ('boy','girl'), ('father','mother'), ('male', 'female') , ('husband','wife'), ('man', 'woman')]
    sampling_group_pairs= [ ('he', 'she'), ('man', 'woman')]
    sampling_group_pairs= [ ('he', 'she'), ('gentlemen', 'ladies'), ('father','mother') , ('him','her'), ('male', 'female')]
    sampling_group_pairs= [ ('he', 'she'), ('man', 'woman'), ('schoolboy','schoolgirl')]
    #sampling_group_pairs= [ ('he', 'she'), ('man', 'woman')]
    num_batches=35
    gamma=0.5

    PLM= LanguageModel(LM_name).to(device)

    actor = ActorNetwork('actor', PLM, device,num_batches,actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias).to(device)
    #actor.model.register_backward_hook(backward_hook)

    # Example of registering the hook on the transformer's embeddings and one of the layers
    #actor.model.transformer.wte.register_full_backward_hook(full_backward_hook)
    #actor.model.transformer.h[0].register_full_backward_hook(full_backward_hook)
    #actor.model.register_full_backward_hook(full_backward_hook)
    reward_obj= Reward( batch_size, adv_batch_size,  PLM, lambda1 , sampling_group_pairs, traits, device)
    #gc.set_debug(gc.DEBUG_LEAK)

    noise_obj= Noise(actor, PLM, reward_obj, sampling_group_pairs, device, batch_size, lambda1, variance_scale_factor, template,num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit)
    print_gpu_usage()
    #actor.update_trainable_layers()
    #print_gpu_usage()
    LA=LearningAgent(batch_size,PLM, actor, noise_obj, lambda1, traits, sampling_group_pairs, device,gamma)
    #print_gpu_usage()
    #compare_models_embeddings(PLM)

    LA.learn(num_batches, reward_obj, noise_obj,PLM,actor)



traits = {'powerless':['weak'],
        'powerful':['capable'],
        'low-status':['inferior'],
        'high-status':['advantaged'],
        'dominated':['dominated'],
        'dominant':['dominant'],
        'poor':['poor'],
        'wealthy':['wealthy'],
        'unconfident':['unconfident'],
        'confident':['confident'],
        'unassertive':['passive'],
        'competitive':['competitive'],
        'traditional':['old-fashioned'],
        'modern':['forward-looking'],
        'religious':['religious'],
        'science-oriented':['logical'],
        'conventional':['conventional'],
        'alternative':['alternative'],
        'conservative':['conservative'],
        'liberal':['liberal'],
        'untrustworthy':['unreliable'],
        'trustworthy':['reliable'],
        'dishonest':['dishonest'],
        'sincere':['sincere'],
        'cold':['unfriendly'],
        'warm':['friendly'],
        'threatening':['threatening'],
        'benevolent':['benevolent'],
        'repellent':['repellent'],
        'likable':['likable'],
        'egotistic':['egotistic'],
        'altruistic':['altruistic']
}



traits = {'powerless':['powerless','weak','helpless','incapable'],
        'powerful':['powerful','forceful','formidable','capable'],
        'low-status':['low-status','unskilled','lowly','inferior'],
        'high-status':['high-status','privileged','elite','advantaged'],
        'dominated':['dominated','subservient','submissive','deferential'],
        'dominant':['dominant','commanding','authoritative'],
        'poor':['poor','impoverished','destitute','needy'],
        'wealthy':['wealthy','affluent','rich','prosperous'],
        'unconfident':['unconfident','bashful','meek','timid'],
        'confident':['confident','self-assured','assured','self-possessed'],
        'unassertive':['unassertive','submissive','diffident','passive'],
        'competitive':['competitive','ambitious','driven','zealous'],
        'traditional':['traditional','old-fashioned'],		
        'modern':['modern','radical','forward-looking'],	
        'religious':['religious','devout','pious','reverent'],
        'science-oriented':['science-oriented','analytical','logical','athiestic'],
        'conventional':['conventional', 'mainstream'],		
        'alternative':['alternative','unorthodox','avante-garde','eccentric'],
        'conservative':['conservative','right-wing','Republican'],	
        'liberal':['liberal','left-wing','Democrat','progressive'],
        'untrustworthy':['untrustworthy','unreliable','undependable'],	
        'trustworthy':['trustworthy','reliable','dependable','truthful'],
        'dishonest':['dishonest','insincere','deceitful'],	
        'sincere':['sincere','genuine','forthright','honest'],
        'cold':['cold','unfriendly','unkind','aloof'],
        'warm':['warm','friendly','kind','loving'],
        'threatening':['threatening','intimidating','menacing','frightening'],
        'benevolent':['benevolent','considerate','generous'],	
        'repellent':['repellent','vile','loathsome','nasty'],
        'likable':['likable','pleasant','amiable','lovable'],
        'egotistic':['egotistic','selfish','self-centered','insensitive'],
        'altruistic':['altruistic','helpful','charitable','selfless']
}



traits = {
    'powerless': ['powerless', 'weak'],
    'powerful': ['powerful', 'capable'],
    'low-status': ['low-status', 'inferior'],
    'high-status': ['high-status', 'advantaged'],
    'dominated': ['dominated', 'submissive'],
    'dominant': ['dominant', 'authoritative'],
    'poor': ['poor', 'needy'],
    'wealthy': ['wealthy', 'prosperous'],
    'unconfident': ['unconfident'],
    'confident': ['confident', 'self-assured'],
    'unassertive': ['unassertive', 'passive'],
    'competitive': ['competitive', 'ambitious'],
    'traditional': ['traditional', 'old-fashioned'],
    'modern': ['modern', 'forward-looking'],
    'religious': ['religious', 'devout', 'pious', 'reverent'],
    'science-oriented': ['science-oriented', 'logical', 'analytical', 'atheistic'],
    'conventional': ['conventional', 'mainstream'],
    'alternative': ['alternative', 'unorthodox', 'avant-garde', 'eccentric'],
    'conservative': ['conservative', 'right-wing', 'Republican'],
    'liberal': ['liberal', 'left-wing', 'Democrat', 'progressive'],
    'untrustworthy': ['untrustworthy', 'unreliable'],
    'trustworthy': ['trustworthy', 'reliable', 'dependable', 'truthful'],
    'dishonest': ['dishonest', 'insincere', 'deceitful'],
    'sincere': ['sincere', 'genuine', 'forthright', 'honest'],
    'cold': ['cold', 'unfriendly', 'unkind', 'aloof'],
    'warm': ['warm', 'friendly', 'kind', 'loving'],
    'threatening': ['threatening', 'intimidating', 'menacing', 'frightening'],
    'benevolent': ['benevolent', 'considerate', 'generous', 'kind'],
    'repellent': ['repellent', 'vile', 'loathsome', 'nasty'],
    'likable': ['likable', 'pleasant', 'amiable', 'lovable'],
    'egotistic': ['egotistic', 'selfish', 'self-centered', 'insensitive'],
    'altruistic': ['altruistic', 'helpful', 'charitable', 'selfless']
}




prior_g_sing = 'person'
prior_g_plur = 'people'
prior_g_cap_plur = 'People'

tplts = {"sing_plur1": ['The <sgroup> is <mask>.']}
groups= {
            "groups_sing" : ['man', 'woman'],
            "groups_plur":['men', 'women'],
            "groups_cap_plur": ['Men', 'Women']
        }



dev=1


    
# def calculate_prob(LM, inputs):
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#     device = torch.device('cuda')

#     # Tokenize traits and convert to IDs in a batch
#     traits_with_prefix = "weak"
#     trait_tokens = tokenizer(traits_with_prefix, return_tensors='pt')
#     trait_ids = trait_tokens['input_ids'].to(device)
    
#     input_ids = tokenizer.encode(inputs, return_tensors='pt').to('cuda')  # Ensure tensors are on the same device as model

#     with torch.no_grad():
#             outputs = LM(input_ids=input_ids)
#             logits = outputs.logits
#     print ("logits.size()",logits.size())
#         # Iterate over the trait_ids and sum their corresponding logits

#     for i, trait_id in enumerate(trait_ids):
#         token_logits = logits[0, -1, :]
#         trait_logit = token_logits[trait_id]  # Get the logit for the current trait_id
#         print ("trait_logit",trait_logit )

#     return trait_logit

# def compare_model_parameters(model, checkpoint):
#     saved_state_dict = checkpoint['model_state_dict']
#     parameters_changed = False

#     for param_name, param in model.named_parameters():
#         print("Checking parameter:", param_name)
#         if param.requires_grad:
#             print("some params requirs grad")
#             # Convert the current parameter to numpy for comparison
#             current_param_np = param.detach().cpu().numpy()

#             # Check if the current parameter exists in the saved state dict and requires_grad
#             if param_name in saved_state_dict:
#                 # Extract the corresponding parameter from the saved state dict
#                 saved_param = saved_state_dict[param_name].detach().cpu().numpy()
#                 #print ( "saved_param",saved_param)
#                 # Compare the current parameter with the saved parameter
#                 if not np.array_equal(current_param_np, saved_param):
#                     print(f"Trainable parameter '{param_name}' has changed.")
#                     parameters_changed = True
#                 else:
#                     print(f"Trainable parameter '{param_name}' remains unchanged.")
#             else:
#                 print(f"Trainable parameter '{param_name}' not found in saved checkpoint.")
#         else:
#             print(f"Parameter '{param_name}' is not set to be trainable.")

#     if not parameters_changed:
#         print("No trainable parameters have changed.")
#     else:
#         print("Some trainable parameters have changed.")

def test_saved_model():
    

    device = torch.device('cuda')
    LM_name = "gpt-2"
    actor_lr=2e-4
    ###actor settings
    in_net=False
    in_net_init_identity=False
    out_net=False
    out_net_init_identity=False
    freeze_ln=False
    freeze_pos=True
    freeze_wte=True
    freeze_ff=False
    freeze_attn=True
    dup_lm_head=False
    dup_lm_head_bias=False

    PLM= LanguageModel('gpt-2')#.to(device)
    #actor =ActorNetwork('actor', PLM, device, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias)#.to(device)

    drive_folder = "/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/"
    checkpoint_filename = "checkpoint2.pth"  # Name of the checkpoint file
    checkpoint_file = os.path.join(drive_folder, checkpoint_filename)


    #device=torch.device('cpu')
    PLM= LanguageModel(LM_name)#.to(device)
    PLM_model=PLM.model
    #lm_head_model =ActorNetwork('actor', PLM, device, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, chkpt_dir).to(device)
    lm_head_model= PLM.model


    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))
    model_state_dict = checkpoint['model_state_dict']

    # Load the state dict into the model
    lm_head_model.load_state_dict(model_state_dict)
    #model_state = lm_head_model.state_dict()

    #compatible_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() }




    # new_state_dict = {}
    # for key, value in checkpoint['model_state_dict'].items():
    #     # Adjust the key name as needed to match the model's expected keys
    #     new_key = key  # Modify this as needed based on your model's architecture
    #     new_state_dict[new_key] = value

    # Now try loading the adjusted state dictionary
    #model_state.update(new_state_dict)
    #lm_head_model.load_state_dict(model_state)


    
    model = lm_head_model.to(device)

    # Print model state dict keys
    print("PLM Model's state dict keys:")
    for key in PLM_model.state_dict().keys():
        print(key)

    # Assuming 'checkpoint' is your loaded checkpoint dictionary
    print("\nCheckpoint's state dict keys:")
    for key in checkpoint['model_state_dict'].keys():
        print(key)

    compare_model_parameters(PLM_model, checkpoint)
    # print( compatible_state_dict.keys())

    inputs= ["The schoolgirl is  ", "The schoolboy is  " , "He is ", "She is"]
    for inp in inputs:
        print (inp, calculate_prob(model, inp) ,calculate_prob(PLM_model,inp) )





# if __name__ == '__main__':

#     print ("omid")
#     test_saved_model()

main()





