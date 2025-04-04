LM_name = "gpt2"
actor_lr=8e-5
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
batch_size= 16
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
sampling_group_pairs= [('he', 'she'), ('man', 'woman'), ('schoolboy','schoolgirl')]
num_batches=16
reward_temp = "the {} is"
num_tokens = 2
bayesian_samples = 100
bayesian_raw_samples = 100
cur_epoch = -1
num_epoch = 1
chkpt_dir = 'ActorModelCheckpoints1/'
num_epochs_LA = 2
lr_drop = 0.2
upsample_quotient = 3

traits = {
    'traditional': ['traditional', 'old-fashioned'],
    'modern': ['modern', 'forward-looking'],
    'religious': ['religious', 'devout', 'pious', 'reverent'],
    'science-oriented': ['science-oriented', 'logical', 'analytical', 'atheistic'],
    'conventional': ['conventional', 'mainstream'],
    'alternative': ['alternative', 'unorthodox', 'avant-garde', 'eccentric'],
    'conservative': ['conservative', 'right-wing', 'Republican'],
    'liberal': ['liberal', 'left-wing', 'Democrat', 'progressive'],
}