from LanguageModel import LanguageModel
from GaussianNoise import Noise, ActorNetwork, Reward
from LearningAgent import LearningAgent
import os
import torch
import sys

print("Please Choose the Model you wish train from the below options:")
print("Enter 1 if you wish to train the Schematune main model (Setting 1)")
print("Enter 2 if you wish to train the Schematune main model (Setting 2)")
print("Enter 3 if you wish to train the Agency only Ablation model")
print("Enter 4 if you wish to train the Belief only Ablation model")
print("Enter 5 if you wish to train the Communion only Ablation model")
print("Enter 6 if you wish to train the He-She only Social group model")
print("Enter 7 if you wish to train the Man-Woman only Social group model")
print("Enter 8 if you wish to train the School boy-School girl only Social group model")

user_input = input('Please enter the number for the model you wish to train: ')

if(user_input == "1"):
    #Main file
    from Parameters.parameters_setting1 import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_setting1 import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_setting1 import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_setting1 import traits, lr_drop, upsample_quotient

    print('Beginning the training of the main model (Setting 1)')

elif(user_input == "2"):
    #Main file
    from Parameters.parameters_setting2 import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_setting2 import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_setting2 import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_setting2 import traits, lr_drop, upsample_quotient

    print('Beginning the training of the main model (Setting 2)')

elif(user_input == "3"):
    #Ablation Agency
    from Parameters.parameters_ablation_agency import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_ablation_agency import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_ablation_agency import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_ablation_agency import traits, lr_drop, upsample_quotient

    print('Beginning the training of the Ablation agency only model')

elif(user_input == "4"):
    #Ablation Belief
    from Parameters.parameters_ablation_belief import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_ablation_belief import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_ablation_belief import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_ablation_belief import traits, lr_drop, upsample_quotient
    
    print('Beginning the training of the Ablation belief only model')

elif(user_input == "5"):
    #Ablation Communion
    from Parameters.parameters_ablation_communion import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_ablation_communion import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_ablation_communion import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_ablation_communion import traits, lr_drop, upsample_quotient

    print('Beginning the training of the Ablation communion only model')

elif(user_input == "6"):
    #Ablation He-She
    from Parameters.parameters_heshe import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_heshe import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_heshe import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_heshe import traits, lr_drop, upsample_quotient

    print('Beginning the training of the Ablation He-She social group only model')

elif(user_input == "7"):
    #Ablation Man-Woman
    from Parameters.parameters_manwoman import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_manwoman import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_manwoman import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_manwoman import traits, lr_drop, upsample_quotient

    print('Beginning the training of the Ablation Man-Woman social group only model')

elif(user_input == "8"):
    #Ablation School boy-School girl
    from Parameters.parameters_schoolboyschoolgirl import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_schoolboyschoolgirl import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_schoolboyschoolgirl import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_schoolboyschoolgirl import traits, lr_drop, upsample_quotient

    print('Beginning the training of the Ablation School boy-School social girl only model')

else:
    #Main file
    from Parameters.parameters_setting2 import LM_name, actor_lr, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, batch_size
    from Parameters.parameters_setting2 import adv_batch_size, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, sampling_group_pairs, num_batches
    from Parameters.parameters_setting2 import reward_temp, num_tokens, bayesian_samples, bayesian_raw_samples, cur_epoch, num_epoch, chkpt_dir, num_epochs_LA
    from Parameters.parameters_setting2 import traits, lr_drop, upsample_quotient
    print('Beginning the training of the main model')

def print_gpu_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    #print(f'Total GPU Memory: {t}, Reserved: {r}, Allocated: {a}, Free: {f}')

def main():
    # logging.warning('This is a warning')
    # logging.info('This is an informational message')
    selected_gpu = sys.argv[0]
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    
    print("")
    print('Training the model with the following parameters: ')
    print(f"Traits: {traits}")
    print(f"Social groups: {sampling_group_pairs}")
    print(f"Learning rate: {actor_lr}, learning rate drop: {lr_drop}")
    print(f"{num_epochs_LA} epochs, Checkpoint directory: {chkpt_dir}, upsampling quotient: {upsample_quotient}")

    PLM = LanguageModel(LM_name).to(device)

    actor = ActorNetwork('actor', PLM, device,num_batches,actor_lr, cur_epoch, num_epoch, in_net, in_net_init_identity, out_net, out_net_init_identity, freeze_ln, freeze_pos, freeze_wte, freeze_ff, freeze_attn, dup_lm_head, dup_lm_head_bias, chkpt_dir, lr_drop).to(device)
    reward_obj= Reward(batch_size, adv_batch_size,  PLM, sampling_group_pairs, traits, device, reward_temp, num_tokens)

    noise_obj= Noise(actor, PLM, reward_obj, sampling_group_pairs, device, batch_size, traits, bayesian_samples, bayesian_raw_samples, variance_scale_factor, template, num_samples, maxiter, improvement_threshold, mixed_sampling_weight, num_restarts, raw_samples, batch_limit, upsample_quotient)
    print_gpu_usage()
    LA=LearningAgent(batch_size,PLM, actor, noise_obj, traits, sampling_group_pairs, device, num_epochs_LA, actor_lr, chkpt_dir)

    LA.learn(num_batches, reward_obj, noise_obj, PLM, actor)

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

    drive_folder = "ActorModelCheckpoints1/"
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

    model = lm_head_model.to(device)

    # Print model state dict keys
    print("PLM Model's state dict keys:")
    for key in PLM_model.state_dict().keys():
        print(key)

    # Assuming 'checkpoint' is your loaded checkpoint dictionary
    print("\nCheckpoint's state dict keys:")
    for key in checkpoint['model_state_dict'].keys():
        print(key)

    LearningAgent.compare_model_parameters(PLM_model, checkpoint)
    # print( compatible_state_dict.keys())

    inputs= ["The schoolgirl is  ", "The schoolboy is  " , "He is ", "She is"]
    for inp in inputs:
        print (inp, LearningAgent.calculate_prob(model, inp) , LearningAgent.calculate_prob(PLM_model,inp) )

main()