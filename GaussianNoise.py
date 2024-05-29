import os
huggingface_token = "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"
os.environ["HF_TOKEN"] = huggingface_token
import logging
import sys
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

import re
import gc
import torch.nn.functional as F

import torch as T

import torch.nn as nn
import torch.optim as optim


import torch.nn as nn



from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch.optim as optim

import os.path

import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
import numpy as np

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
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
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.sampling import SobolQMCNormalSampler
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP

#import warnings
from botorch.acquisition import qNoisyExpectedImprovement

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize

from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

from botorch.utils.transforms import unnormalize

from botorch.models.transforms.input import Normalize

from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP
from botorch.fit import fit_gpytorch_model

import os
os.environ['OMP_NUM_THREADS'] = '1' # speed up

#from DGEMO.problems.common import build_problem
#from DGEMO.mobo.algorithms import get_algorithm
#from DGEMO.visualization.data_export import DataExport
#from DGEMO.arguments import get_args
#from DGEMO.utils import save_args, setup_logger
import gc

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

class Noise:
    def __init__(self, actor, PLM, reward_obj, sampling_group_pairs, device, batch_size, traits, bayesian_samples, bayesian_raw_samples, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight,num_restarts,
                    raw_samples,batch_limit,upsample_quotient):
        #self.lambda1=lambda1
        self.num_tokens=2
        self.num_samples=num_samples
        self.maxiter=maxiter
        self.improvement_threshold=improvement_threshold
        self.mixed_sampling_weight=mixed_sampling_weight
        self.num_restarts=num_restarts
        self.raw_samples=raw_samples
        self.batch_limit=batch_limit

        self.variance_scale_factor=variance_scale_factor #0.1
        self.traits=traits
        self.batch_size = batch_size
        self.tokenizer = actor.tokenizer
        self.device = device #T.device('cuda')
        self.upsample_quotient = upsample_quotient
        self.embedding_dim= actor.input_dim
        self.sampling_group_pairs= sampling_group_pairs #[('man', 'woman'), ('stepfather', 'stepmother')]
        self.num_groups= len(self.sampling_group_pairs)
        self.num_groups_per_pair = len(self.sampling_group_pairs[0])  #Assuming each tuple in group_pairs has the same number of elements
        self.template = template #"the {} is powerful."
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
        self.defining_group_sets_1 = [ ('man', 'woman'), ('he', 'she'), ('father', 'mother'), ('boy', 'girl'),('groom', 'bride'), ('brother', 'sister')]
        self.U = self.bias_subspace(actor)

        self.bias_subspace_dim = len(self.sampling_group_pairs)
        self.samples=[]
        self.default_values = self.initialize_data()

        self.start_cond=True
        self.mu_prime_Batch=torch.zeros((self.batch_size, self.num_groups_per_pair, self.num_groups,self.num_tokens, self.embedding_dim))
        self.reward = reward_obj
        self.device_cpu = torch.device("cpu")
        self.model=None
        self.bayesian_samples = bayesian_samples
        self.bayesian_raw_samples = bayesian_raw_samples

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

              # Add noise in the direction of normalized difference
              norm_diff = diff / torch.norm(diff, dim=-1, keepdim=True)
              perturbed_embeddings = norm_diff #+ gaussian_noise

              # Store perturbed embeddings
              U[i] = perturbed_embeddings.unsqueeze(0)

      print("this is U", U)
      return U

    def get_variances(self):
        variances = self.S[:]  # Variance along each principal component
        variances_list = variances.tolist()  # Convert tensor to a list of numbers
        return variances_list

    def set_bounds(self):
        bounds = []
        print ("self.variances",self.variances)
        for i, var in enumerate(self.variances):
            std_scale = np.sqrt(var) * 0.3 #self.variance_scale_factor

            # Bounds for the mean value
            mean_lower_bound =  - std_scale
            mean_upper_bound =  std_scale
            bounds.append((mean_lower_bound, mean_upper_bound))

        return torch.tensor(bounds, requires_grad=True).T

    def gaussian_noise_subspace(self, init_x):
        # Convert bias_subspace from numpy array to PyTorch tensor
        bias_subspace_tensor = self.U.to(self.device)
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
                    
                    # Assign the accumulated noise to the corresponding position in the noise tensor
                    noise_tensor[b, g,g_pair, :, :] = temp_noise

        del bias_subspace_tensor

        return noise_tensor

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
            
            #flattened_mu = self.normalize_batch_mu(mu_prime_B)
            #mu_prime_B = flattened_mu.to(self.device)
            
            reward = self.reward.calculate_reward(mu_prime_B.detach(), actor, PLM).detach()

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
        reward = self.reward.calculate_reward(mu_prime_B.detach(), actor, PLM, True, True).detach()
        
        batch_rewards = reward
        
        # Convert the final PyTorch tensor back to a NumPy array with the same dimensionality
        final_result = (-1 / batch_rewards).view(-1, 2).cpu().numpy()
        
        return final_result

    # this is the orginal run bayesian optimization fnction
    def run_bayesian_optimization(self, actor, PLM, candid):
        best_value = float('-inf')
        self.samples = []
        #total_num_tasks = 1 #self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups
        num_samples=self.bayesian_samples
        #maxiter=5
        #improvement_threshold=0.0001
        #mixed_sampling_weight=0.2
        num_restarts=self.num_restarts
        raw_samples=self.bayesian_raw_samples
        #batch_limit=8

        print ("entering the function bo")
        print_gpu_usage()
        #upsampleq=3
        upsampleq=self.upsample_quotient
        gc.collect()
        torch.cuda.empty_cache()
        device_cpu = torch.device("cpu")
        N_W = raw_samples
        STD_DEV = 0.05
        ALPHA = 0.8
        tkwargs = {"device": "cpu", "dtype": torch.double}
        #likelihood = GaussianLikelihood()
        lower_bound = -1.0
        upper_bound = 1.0

        # Create a tensor with the bounds for each dimension
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

            with torch.no_grad():

                init_y = self.problem(init_x, actor, PLM).detach().double()

            init_y = init_y.view(init_x.size(0), 1)  # Ensure init_y is double
            print(init_x.size(), init_y.size(), "final")

            init_y=init_y.to(device_cpu )
            
            init_x_flat = init_x.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).detach().double()  # Ensure init_x_flat is double
            task_feature=init_x_flat.shape[-1] - 1
            #likelihood = MultitaskGaussianLikelihood(num_tasks=total_num_tasks)

            inpf=Normalize(d=init_x_flat.size(1))

            if self.model== None:
                self.model = SingleTaskGP(init_x_flat, init_y,input_transform=inpf, outcome_transform=Standardize(m=1))
                self.model.double()  # Ensure the model parameters are double
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                fit_gpytorch_model(mll)

            self.X_all = torch.cat([self.X_all, init_x_flat], dim=0)
            self.Y_all= self.Y_all.to(device_cpu)
            self.Y_all = torch.cat([self.Y_all, init_y.detach()], dim=0)#.view(-1, 1)
            
            print ("sizes", init_x_flat.size(), init_y.size(), self.X_all.size(), self.Y_all.size())

            self.best_value = best_value
            self.start_cond = False

        else:
            init_x = candid.double()  # Ensure candid is double
            candid = candid.to('cpu')

            # Ensure init_y is double

            init_x_flat = init_x.view(-1, self.bias_subspace_dim * self.num_groups_per_pair * self.num_groups).double()  # Ensure init_x_flat is double
            # init_y = self.problem(init_x_flat, actor, PLM).double()
            # init_y= standardize(init_y + 0.05 * torch.randn_like(init_y))
            # #init_y = standardize(init_y + 1e-1 * torch.randn_like(init_y))

        d = init_x_flat.shape[-1]

        self.model.to(device_cpu )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))

        acqf = qNoisyExpectedImprovement(
            model=self.model,
            X_baseline=self.X_all,
            sampler=sampler,
            #   objective=risk_measure,
            prune_baseline=True,
        )

        print ("finished acq")
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds_tensor,
            q=self.batch_size*upsampleq,
            num_restarts=num_restarts,
            raw_samples=min (self.X_all.size(0), raw_samples),
        )

        candidates = candidate.detach()  # Detach once here
        torch.cuda.empty_cache()
        new_y_values = []

        candidate_reshaped = candidates.view(self.batch_size * upsampleq,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)

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
        #new_y_flat = self.problem(self.candidates.detach(), actor, PLM).detach()
                    # Assuming new_y_flat contains the objective values for each candidate and you want to maximize these values
        sorted_indices = torch.argsort(new_y_flat, descending=True)  # Change to ascending=False if you want to minimize

        # Select the top batch_size candidates
        candidates=candidates.to(device_cpu)
        best_indices = sorted_indices[:self.batch_size]
        best_indices=best_indices.to(device_cpu)
        best_mu = new_mu_flat[best_indices].squeeze(dim=1)

        best_mu = best_mu.to(device_cpu)

        self.best_X = candidates[best_indices]
        print ("self.best_X.shape",self.best_X.shape)

        new_y_flat = new_y_flat.to(device_cpu).view(upsampleq*self.batch_size,1)

        self.best_Y = new_y_flat[best_indices]

        # Detach and reshape for consistency
        self.best_X = self.best_X #.detach().clone()
        self.best_Y = self.best_Y.detach().clone().view(-1).squeeze(-1)

        # Update X_all and Y_all with all candidates and their corresponding y values
        self.X_all=self.X_all.clone()
        self.X_all = torch.cat([self.X_all, candidate_unnormalized.detach()], dim=0)

        self.Y_all = torch.cat([self.Y_all.clone(), new_y_flat.detach()], dim=0)
        self.Y_all = self.Y_all#.view(-1,1)

        self.X_all = self.X_all#.detach()
        self.Y_all = self.Y_all#.detach()
        self.X_all = self.X_all.double()
        self.Y_all = self.Y_all.double()

        # Set the new training data for the model
        self.model.double()  # Ensure the model parameters are double
        # Reshape Y from [n] to [n, 1]
        # intf = InputPerturbation(
        #     perturbation_set=draw_sobol_normal_samples(d=self.X_all.shape[-1], n=min ( num_samples,self.Y_all.size(0)), **tkwargs) * STD_DEV,
        #     bounds=bounds_tensor,
        #     )

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

        self.model = SingleTaskGP(self.X_all, self.Y_all, input_transform=inpf, outcome_transform=Standardize(m=1))
        self.model.set_train_data(inputs=self.X_all, targets=self.Y_all.clone().squeeze(-1), strict=False)

        # Optimize the model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

        # Process the entire batch of best candidates

        self.best_X =unnormalize(self.best_X, bounds_tensor).detach().clone()#.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
        #best_X_reshaped= self.best_X.view(self.batch_size,self.bias_subspace_dim, self.num_groups_per_pair, self.num_groups)
        #noise_tensor = self.gaussian_noise_subspace(best_X_reshaped)
        #self.mu_prime_Batch = self.calculate_noisy_embeddings(self.group_embeddings, noise_tensor, True).to(self.device)
        self.mu_prime_Batch = self.calculate_noisy_embeddings(self.group_embeddings.detach(), best_mu, True).to(self.device)
        del candidates, batch_candidates, batch_y, new_y_flat, new_y_list
        torch.cuda.empty_cache()
        gc.collect()
        return self.best_X

    def plot_reward_distribution(self):
        # Ensure Y_all is a tensor and then flatten it
        Y_all_flattened = self.Y_all.view(-1).squeeze(-1).flatten()

        # Convert each tensor element to a scalar
        scalar_rewards = [r.item() for r in Y_all_flattened]

        rewards_array = np.array(scalar_rewards)
        num_bins = int(np.sqrt(len(rewards_array)))

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
            outputs = actor.forward_wte( sentence)#.squeeze(0)
        
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

        if self.start_cond==True:
            candidates=self.default_values
        else:
            candidates= self.best_X

        # Run Bayesian optimization and get the noise mean value
        bo_results = self.run_bayesian_optimization(actor, PLM, candidates)

    def sample_batch(self, actor, PLM, checknoise=True):
        group_embeddings = self.calculate_embeddings(actor).to(self.device) #T.zeros((self.num_groups, self.num_groups_per_pair,self.num_tokens, self.embedding_dim))  # Initialize tensor for embeddings

        # Calculate noisy base embedding
        if checknoise==True:
          self.sample(actor, PLM)
          noisy_base_embedding = self.calculate_noisy_embeddings(group_embeddings.detach(), self.mu_prime_Batch.to(self.device), checknoise).to(self.device)

          return noisy_base_embedding, group_embeddings

        else:
          return group_embeddings

class ActorNetwork(nn.Module,metaclass=SingletonType):
    # no need for additional_layer, name
    def __init__(self, network, language_model,device,num_batches,lr, cur_epoch, num_epoch,
         in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos,
                              freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, chkpt_dir, lr_drop):

        super(ActorNetwork, self).__init__()
        self.num_epochs = num_epoch #1
        self.lr= lr
        self.current_epoch = cur_epoch #-1  # Start from epoch ?
        self.tokenizer= language_model.tokenizer
        self.device = device
        self.num_batches = num_batches
        self.dup_lm_head= dup_lm_head
        self.checkpoint_file= chkpt_dir+'checkpoint2.pth'
        in_layer_sizes = []
        out_layer_sizes = []
        self.in_net= in_net
        self.out_net=out_net
        self.model = language_model.model.to(self.device)
        self.input_dim = language_model.model.config.n_embd
        self.dropout = lr_drop
        orth_gain = 1.41
        in_net_init_identity = True
        self.freeze_ln=freeze_ln
        self.freeze_wte= freeze_wte
        self.freeze_pos=freeze_pos
        self.total_layers = len(self.model.transformer.h)
        target_parameters = 0

        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad], lr=self.lr
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches)

    def increment_epoch(self):
        self.current_epoch += 1
        self.update_trainable_layers()

    def update_trainable_layers(self):
        total_layers = self.total_layers #len(self.model.transformer.h)  # Total number of transformer blocks in the model
        self.current_epoch += 1
        
        # Define the start layer from which you want to unfreeze the layer normalization parameters
        start_unfreeze_layer = max(0, total_layers - 4 - self.current_epoch)
        print ("start_unfreeze_layer",start_unfreeze_layer)

        for name, param in self.model.named_parameters():
            # Initially freeze all parameters
            param.requires_grad = False
            
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

        base_params = [param for name, param in self.model.named_parameters() if 'h.11' not in name and param.requires_grad]
        lm_head_params = [param for name, param in self.model.named_parameters() if 'ln.f' in name and param.requires_grad]

        # Now, set up the optimizer with different learning rates
        if base_params or lm_head_params:
            self.optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': self.lr},  # Standard learning rate for base model parameters
                #{'params': lm_head_params, 'lr': self.lr*0.1}  # Adjusted learning rate for LM head parameters
                {'params': lm_head_params, 'lr': self.lr*self.dropout}
            ])
        else:
            raise ValueError("No parameters with requires_grad=True. Check your model's parameter setup.")

        # Setup the scheduler with the optimizer
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches)

    def forward_text(self, text): # this forward's output is aka state
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
        outputs = self.model.transformer(inputs_embeds=action)

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

    def resize_token_embeddings(self, new_num_tokens=None):
        if new_num_tokens is None:
            new_num_tokens = len(self.tokenizer)
        self.core_model.resize_token_embeddings(new_num_tokens)

    def save_checkpoint(self, checkpoint_name="checkpoint2.pth"):
        # Specify the folder in Google Drive to save the checkpoint
        drive_folder = "ActorModelCheckpoints1/"
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
                self.iteration += 1
            else:
                # Update running means for L and loss ratios l
                self.running_mean_L[i] = decay * self.running_mean_L[i-1] + (1 - decay) * loss.mean().unsqueeze(0)
                loss_ratio = loss / self.running_mean_L[i]
                self.running_mean_l[i] = decay * self.running_mean_l[i-1] + (1 - decay) * loss_ratio.mean().unsqueeze(0)

                # Update variances for L and l
                delta_L = loss - self.running_mean_L[i]
                delta_l = loss_ratio - self.running_mean_l[i]
                self.running_var_L[i] = decay * self.running_var_L[i-1] + (1 - decay) * (delta_L ** 2).mean().unsqueeze(0)
                self.running_var_l[i] = decay * self.running_var_l[i-1] + (1 - decay) * (delta_l ** 2).mean().unsqueeze(0)

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
    def __init__(self, Bigbatch_size, batch_size, PLM, sampling_group_pairs, traits, device, reward_temp, num_tokens):
        self.device = device # T.device("cuda")
        #self.lambda1  = lambda1
        self.num_iterations=0
        #self.tau = tau
        self.batch_size = batch_size
        self.Bigbatch_size = Bigbatch_size
        self.garbage_dict= {}
        self.garbage_dict_count=0
        self.num_tokens=num_tokens
        self.reward= None

        self.sampling_group_pairs=sampling_group_pairs
        self.traits= traits
        self.embedding_dim= PLM.d_model
        self.sampling_group_pairs=sampling_group_pairs #[ ('man', 'woman'),('stepfather', 'stepmother')]
        self.num_groups= len(self.sampling_group_pairs)
        self.num_groups_per_pair = len(self.sampling_group_pairs[0])  # Assuming each tuple in group_pairs has the same number of elements
        self.template = reward_temp #"The {} is "
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

    def calculate_reward(self, mu_prime, LM, PLM, weighted=False, both= False):

        #mu_prime, _= self.noise.sample_batch (LM)
        batchNoiseEval= self.NoiseEval( LM, mu_prime, self.sampling_group_pairs, self.traits, None, "The <group> is <mask>")
        
        if batchNoiseEval.size()[0] == self.batch_size:
            prime=self.prime1

        else:
            prime= self.primeB

        klG1, klG2= self.kl_perf( LM, PLM, mu_prime, prime, self.sampling_group_pairs, "The <group> is powerful")

        fairness_coefficient=20
        
        epsilon = 1e-6
        
        #print("combined_reward",combined_reward)
        self.NoiseE.append(batchNoiseEval.mean(0))
        self.klp.append((klG1 + klG2).mean(0)*1)
        print ("batchNoiseEvalmagnified ,", batchNoiseEval ,"lambda1*(klG1+ klG2):",  1*(klG1+ klG2))
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
        losses_tensor= torch.tensor([torch.mean(batchNoiseEval), torch.mean (-1* (klG1 + klG2))])
        _, weights = self.CoVWeighting.adjust_loss_weights(losses_tensor)
        
        adjusted_reward= batchNoiseEval*weights[0] - 1* (klG1 + klG2)*weights[1]
        
        if both == False:
            return adjusted_reward
        else:
            return torch.stack((batchNoiseEval*weights[0], - 1* (klG1 + klG2)*weights[1]+ batchNoiseEval*weights[0]), dim=1)

        #return combined_reward
        #return batchNoiseEval*20

    def calculate_embedding(self,actor, sentence: str):
        #inputs = self.tokenizer(sentence, return_tensors='pt')
        with T.no_grad():
            outputs = actor.forward_wte( sentence)#.squeeze(0)

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

        device = self.device #torch.device(f'cuda:{selected_gpu}')

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

    def calculate_logits(self, LM, tmplt, mu_prime_g, inputs, group):
        selected_gpu = sys.argv[1]

        device = self.device #torch.device(f'cuda:{selected_gpu}')

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

        device = self.device #torch.device(f'cuda:{selected_gpu}')

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
    
    def NoiseEval(self, LM, mu_prime, sampling_group_pairs, traits, PLM=None, tmplt="The <group> is <mask>"):
        batch_size = mu_prime.size(0)

        # Assuming dimensions for 'sampling_group_pairs' and 'traits' are known or can be calculated
        num_groups = len(sampling_group_pairs) * 2  # Assuming each pair has 2 groups
        num_traits = sum(len(ts) for ts in traits.values())

        # Pre-allocate tensors for KL divergences, initialized to zeros
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
                    inputs = LM.tokenizer(input_txt, return_tensors='pt')

                    selected_gpu = sys.argv[1]

                    #inputs = {key: value.to(torch.device(f'cuda:{selected_gpu}')) for key, value in inputs.items()}
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}

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
            concatenated_p2_softmax = F.softmax(concatenated_p2, dim=0)
            concatenated_p1_log_softmax = F.log_softmax(concatenated_p1, dim=0)

            # Compute KL divergence on the softmaxed distributions
            kl_div = F.kl_div(concatenated_p1_log_softmax, concatenated_p2_softmax, reduction='mean')

            num_elements = len(sampling_group_pairs)  # Or any other appropriate count that reflects your data structure
            kl_div_average = kl_div / num_elements

            kl_div_averages_list.append(kl_div_average)

        # Step 4: Stack the list into a tensor after the loop
        self.kl_divs_tensor = torch.stack(kl_div_averages_list)
        self.kl_divs_tensor.to(self.device)
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
      token_probs_1 = last_token_logits_1
      token_probs_2 = last_token_logits_2

      # Retrieve and update probabilities for complementary tokens
      for token in complementary_tokens_1:
          token_id = LM.tokenizer.encode(token)[0]  # Ensure no special tokens are added
          token_probs_dict_1[token] = token_probs_1[token_id]  # Removed .item() to keep tensor

      for token in complementary_tokens_2:
          token_id = LM.tokenizer.encode(token)[0]  # Ensure no special tokens are added
          token_probs_dict_2[token] = token_probs_2[token_id]  # Removed .item() to keep tensor

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

        klG1 = kl_divergence_group1.mean(dim=1)  # Mean across group pairs
        klG2 = kl_divergence_group2.mean(dim=1)

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