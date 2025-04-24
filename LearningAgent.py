import os
huggingface_token = "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"
os.environ["HF_TOKEN"] = huggingface_token
import copy
import math
import logging
import sys
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
from Evaluation.Crows_StereoSet import evaluate
from Evaluation.StereoSet_FinalScores import get_scores

from torch.cuda.amp import GradScaler, autocast
from LanguageModel import LanguageModel
from GaussianNoise import ActorNetwork
import re
import gc
import torch.nn.functional as F

import torch as T

import torch.nn as nn
import torch.optim as optim

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel

import os.path

import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
import numpy as np

import matplotlib.pyplot as plt
#import warnings

import matplotlib.pyplot as plt

import os
os.environ['OMP_NUM_THREADS'] = '1' # speed up
import numpy as np
#from DGEMO.problems.common import build_problem
#from DGEMO.mobo.algorithms import get_algorithm
#from DGEMO.visualization.data_export import DataExport
#from DGEMO.arguments import get_args
#from DGEMO.utils import save_args, setup_logger
import gc

class LearningAgent(object):
    def __init__(self, batch_size,PLM, actor, noise_obj, traits, sampling_group_pairs, device, num_epochs, actor_lr, chkpt_dir):
        self.chkpt_path = chkpt_dir #"/home/oshokrol/zero-shot-2/ActorModelCheckpoints1/checkpoint2.pth"
        self.device = device#T.device('cuda')
        #self.lambda1  = lambda1
        #self.tau = tau
        self.batch_size = batch_size
        self.reward = None
        #self.actor = actor # ActorNetwork( 'Actor', PLM)
        self.noise = noise_obj
        self.traits= traits
        self.num_traits = len(traits)  # Number of Traits
        #self.gamma=gamma
        self.old_actor=PLM
        #self.clip_param=clip_param
        self.num_epochs=num_epochs
        self.actor_lr=actor_lr

        self.embedding_size = actor.model.config.n_embd
        self.sampling_group_pairs=sampling_group_pairs# [('man', 'woman') , ('stepfather', 'stepmother')]
        self.pairs_count = len (self.sampling_group_pairs)
        self.pair_size = len (self.sampling_group_pairs[0])
        self.trait_count = len(traits.keys())
        self.template = "The {group} is {trait}" #learning_template #"The {} is "
        self.Policy_Loss = []
        self.start_save=None
        self.garbage_dict_count = 0
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

    def calculate_advantage_zero_noise(self, noise_obj, reward_obj):
        return 0

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
        selected_gpu = sys.argv[0]
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(dev) #torch.device(f'cuda:{selected_gpu}')

        # Tokenize traits and convert to IDs in a batch
        traits_with_prefix = "calm"
        trait_tokens = tokenizer(traits_with_prefix, return_tensors='pt')
        trait_ids = trait_tokens['input_ids'].to(device)
        
        #input_ids = tokenizer.encode(inputs, return_tensors='pt').to(torch.device(f'cuda:{selected_gpu}'))  # Ensure tensors are on the same device as model
        input_ids = tokenizer.encode(inputs, return_tensors='pt').to(device)  # Ensure tensors are on the same device as model

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

        selected_gpu = sys.argv[0]

        checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))

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

        inputs= ["When I was a schoolboy, I was always ", "When I was a schoolgirl, I was always BLANK" , "He was alway ", "She was always "]
        for inp in inputs:
            print (inp, self.calculate_prob(model, inp) ,self.calculate_prob(PLM_model,inp) )

    def ppo_update(self, samples, noise_obj, reward_obj, PLM,actor, epoch):
        (mu_prime_Batch)=samples
        #batch_rewards = reward_obj.calculate_reward(mu_prime_Batch, actor, PLM, self.lambda1,True)
        batch_rewards = reward_obj.calculate_reward(mu_prime_Batch, actor, PLM)
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
        #print ("policy loss magnified by 100", policy_loss)
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
        plt.xlabel('Bayesian Optimization Iteration')  # X-axis label
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
        plt.title('KL performance average Trend per Iteration')  # Title of the plot
        plt.xlabel('Bayesian Optimization Iteration')  # X-axis label
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
        plt.title('Bias Evaluation average Trend per Iteration')  # Title of the plot
        plt.xlabel('Bayesian Optimization Iteration')  # X-axis label
        plt.ylabel('Bias')  # Y-axis label
        plt.grid(True)  # Show grid

        # Optional: If you want to see a smoother trend, consider plotting a moving average
        window_size = 50  # Define the window size for the moving average
        moving_avg = [sum(NoiseEval[i:i+window_size])/window_size for i in range(len(NoiseEval)-window_size+1)]
        plt.plot(range(window_size-1, len(NoiseEval)), moving_avg, linestyle='-', color='r', label='Moving Average')  # Plot moving average
        unique_plot3_filename = self.unique_filename('plot3', '.png')
        plt.savefig(unique_plot3_filename)
        plt.legend()
        plt.show()

    def save_checkpoint(self, actor_model, epoch, base_path='ActorModelCheckpoints1'):
        """
        Save the model checkpoint at the end of an epoch.
        
        Parameters:
        - actor_model: The model to save.
        - epoch: The current epoch number.
        - base_path: Base directory to save checkpoints.
        """
        # Ensure the base directory exists
        #os.makedirs(base_path, exist_ok=True)
        os.makedirs(self.chkpt_path, exist_ok=True)

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
        #filepath = os.path.join(base_path, filename)
        filepath = os.path.join(self.chkpt_path, filename)

        # Save the model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': actor_model.state_dict(),
        }, filepath)

        print(f"The Checkpoint is saved at this path: {filepath}")


    def learn(self, num_batches, reward_obj, noise_obj, PLM, actor):

        T_max = 100  # Example value, adjust as needed
        # Generate 8 log-linearly spaced learning rates between 1e-4 and 3e-4
        #learning_rates = np.logspace(np.log10(1e-4), np.log10(3e-4), num=1)
        actor.num_epochs= self.num_epochs
        coef= 1
        #actor.increment_epoch()
        actor.update_trainable_layers()

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

            #actor.increment_epoch()
            actor.update_trainable_layers()
            
            if (epoch>1 and epoch<=3):
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
            #overall_results, intrasentence_bias, intersentence_bias = eval_stereoset(model_copy, model_copy, True)
            
            evaluate(self.device, "0", "ActorModelCheckpoints1/checkpoint2.pth", "crows", "gender", 42,  "./results_schematune/")
            evaluate(self.device, "0", "ActorModelCheckpoints1/checkpoint2.pth", "crows", "race", 42,  "./results_schematune/")
            evaluate(self.device, "0", "ActorModelCheckpoints1/checkpoint2.pth", "crows", "religion", 42,  "./results_schematune/")
            evaluate(self.device, "0", "ActorModelCheckpoints1/checkpoint2.pth", "stereoset", "religion", 42,  "./results_schematune/")            
            get_scores() #Prints the final StereoSet scores

            del model_copy

            noise_obj.start_condition= True

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