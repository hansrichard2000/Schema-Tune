import os
import json
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import GPT2Tokenizer
import torch

import transformers
from transformers import AutoConfig,AutoTokenizer
from datasets import load_dataset

#from .arguments import get_args
#from model.utils import get_model
from dataset.language_modeling import get_tokenized_datasets
from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias
#from bias_bench.model import models

from transformers import TrainingArguments
from arguments import ModelArguments, DataTrainingArguments  # Adjust the import based on your project structure
from Evaluation.StereoSet_FinalScores import get_scores

def evaluate(device, cudaNum="0", model_file_name="ActorModelCheckpoints/checkpoint.pth", datasetname="stereoset",bias_type="gender",seed=42, outputdir=''):


	# Assuming get_args() has been called and returned model_args, data_args, training_args

	# Manually override arguments
	# Example based on your first bash command

	# Make sure to select the correct device
	os.environ["CUDA_VISIBLE_DEVICES"] ="0" # "0"  # Set CUDA device

	# The following part depends on how you implement or use `evaluate.py` and `stereoset_evaluation.py` in your script.
	# For simplicity, you might directly call the functions that these scripts would trigger,
	# passing the above arguments as function parameters or setting them globally as shown.


	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	from transformers import GPT2Config

	# Define the configuration
	config = GPT2Config(
		vocab_size=50257,  # Set the size of the vocabulary
		n_positions=1024,
		n_ctx=1024,
		n_embd=768,
		n_layer=12,
		n_head=12,
		# Add any other model-specific parameters here
	)

	# Save the configuration to disk
	config.save_pretrained('ActorModelCheckpoints/')


	model_args = ModelArguments(
    	model_name_or_path=model_file_name,
        # Add other necessary parameters here
    )
	data_args = DataTrainingArguments(
        dataset_name=datasetname,
        bias_type=bias_type,
        # Add other necessary parameters here
    )
	training_args = TrainingArguments(
        output_dir=outputdir,
        seed=seed,
        # Add other necessary parameters here
    )
	#model_args,data_args,training_args = get_args()
	model_args.task_type = "causal_lm"

	# model_args.model_name_or_path = model_file_name # "ActorModelCheckpoints/checkpoint.pth"
	# data_args.dataset_name = dataset_name#"stereoset"  # Change this line for each dataset
	# data_args.bias_type = bias_type#"gender"
	# training_args.seed = seed#42
	# training_args.output_dir = outputdir#''  # Ensure you handle empty strings appropriately in your script


	transformers.set_seed(seed)

	# Load config and tokenizer
	config_kwargs = {
		"cache_dir": model_args.cache_dir,
		"revision": model_args.model_revision,
		"use_auth_token": True if model_args.use_auth_token else None,
	}
	if model_args.config_name:
		config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
	elif model_args.model_name_or_path:
		#config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
		config = AutoConfig.from_pretrained("gpt2")
	else:
		config = CONFIG_MAPPING[model_args.model_type]()
		logger.warning("You are instantiating a new config instance from scratch.")
		if model_args.config_overrides is not None:
			logger.info(f"Overriding config: {model_args.config_overrides}")
			config.update_from_string(model_args.config_overrides)

	tokenizer_kwargs = {
		"cache_dir": model_args.cache_dir,
		"use_fast": model_args.use_fast_tokenizer,
		"revision": model_args.model_revision,
		"use_auth_token": True if model_args.use_auth_token else None,
	}
	if model_args.tokenizer_name:
		tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
	elif model_args.model_name_or_path:
		#tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
		#tokenizer = AutoTokenizer.from_pretrained("ActorModelCheckpoints/config.json", **tokenizer_kwargs)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

	else:
		raise ValueError(
			"You are instantiating a new tokenizer from scratch. This is not supported by this script."
			"You can do it from another script, save it, and load it from here, using --tokenizer_name.")

	# Set padding token.
	if model_args.task_type=="causal_lm":
		tokenizer.pad_token = tokenizer.eos_token
		config.pad_token_id = config.eos_token_id

	# Load model
	bias_bench_models = ["SentenceDebiasBertForMaskedLM","INLPBertForMaskedLM","SelfDebiasBertForMaskedLM",
		"SentenceDebiasGPT2LMHeadModel","INLPGPT2LMHeadModel","SelfDebiasGPT2LMHeadModel"]
	if model_args.prompt_model in bias_bench_models:
		debiased_model_to_base_model = {
			"SentenceDebiasBertForMaskedLM":'BertModel',
			"INLPBertForMaskedLM":'BertModel',
			"SelfDebiasBertForMaskedLM":'BertModel',
			"SentenceDebiasGPT2LMHeadModel":'GPT2Model',
			"INLPGPT2LMHeadModel":'GPT2Model',
			"SelfDebiasGPT2LMHeadModel":'GPT2Model'}
		kwargs = {}
		if 'SentenceDebias' in model_args.prompt_model:
			bias_direction = "results/subspace/subspace_m-{}_c-{}_t-{}.pt".format(
				debiased_model_to_base_model[model_args.prompt_model],model_args.model_name_or_path,data_args.bias_type)
			kwargs["bias_direction"] = torch.load(bias_direction)
		if 'INLP' in model_args.prompt_model:
			projection_matrix = "results/projection_matrix/projection_m-{}_c-{}_t-{}_s-0.pt".format(
				debiased_model_to_base_model[model_args.prompt_model],model_args.model_name_or_path,data_args.bias_type)
			kwargs["projection_matrix"] = torch.load(projection_matrix)
		model = getattr(models, model_args.prompt_model)(model_args.model_name_or_path, **kwargs)
		if _is_self_debias(model_args.prompt_model):
			model._model.eval()
			model._model.to(device)
		else:
			model.eval()
			model.to(device)
	else:
		if model_args.prefix_tokens is not None:
			model_args.prefix_tokens = tokenizer.encode(model_args.prefix_tokens,add_special_tokens=False)
			print('use real word for initialization, prefix length: {}'.format(len(model_args.prefix_tokens)))
		#model = get_model(model_args,config)
		# note that for evaluation, `model_args.model_name_or_path` should be set to the checkpoints saved by debias_xxx.py
		#model.resize_token_embeddings(len(tokenizer))
		#model.to(device)
		#model.eval()


	model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
	checkpoint = torch.load(model_file_name, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'], strict=False)
	model.to(device)

	class_name = model_args.prompt_model if model_args.prompt_model in bias_bench_models else model.__class__.__name__
	if data_args.dataset_name=='crows':
		runner = CrowSPairsRunner(
			model=model,
			tokenizer=tokenizer,
			input_file=os.path.join('data','crows','crows_pairs_anonymized.csv'),
			bias_type=data_args.bias_type,
			is_generative=_is_generative(class_name),  # Affects model scoring.
			is_self_debias=_is_self_debias(class_name),
		)
		results = runner() # a number
		print(f"Metric: {results}")
	
	elif data_args.dataset_name=='stereoset':
		runner = StereoSetRunner(
			intrasentence_model=model,
			tokenizer=tokenizer,
			input_file=os.path.join('data','stereoset','test.json'),
			model_name_or_path=model_args.model_name_or_path,
			batch_size=1, # training_args.per_device_eval_batch_size,
			is_generative=_is_generative(class_name),
			is_self_debias=_is_self_debias(class_name),
			bias_type='race-color' if data_args.bias_type=='race' else data_args.bias_type,
		)
		results = runner() # a nested dict
		#print(f"Metric: {results}")

	print (results)
	os.makedirs(outputdir, exist_ok=True)

    # Define the path to the results file
	results_file_path = os.path.join(outputdir, f"{datasetname}_results.json")
	
	# Save the results to the file
	with open(results_file_path, "w") as f:
		json.dump(results, f, indent=2)


#evaluate( "cuda:0", "0", "ActorModelCheckpoints1/checkpoint_epoch1_exp0.pth", "stereoset", "gender", 42, './results_schematune/')
#get_scores()
#evaluate( "cuda:0", "0", "ActorModelCheckpoints1/checkpoint_epoch1_exp0.pth", "crows", "gender", 42,  "./results_schematune/")
#evaluate( "cuda:0", "0", "ActorModelCheckpoints1/checkpoint_epoch1_exp0.pth", "crows", "race", 42,  "./results_schematune/")
#religion = evaluate( "cuda:0", "0", "ActorModelCheckpoints1/checkpoint_epoch1_exp0.pth", "crows", "religion", 42,  "./results_schematune/")