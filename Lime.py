'''
sys.argv[1] : cluster data
sys.argv[2] : vocubulary (part of the best model parameters)
sys.argv[3] : best ConvAE model

'''
import sys
import torch
import pandas as pd
import numpy as np
import os 
from collections import defaultdict

from lime.lime_text import LimeTextExplainer


sys.path.append('...') #convAE code
import utils as ut
import evaluate

sys.path.append('...') #convAE code
import net
import data_loader


#best convAE model
model_param = ut.model_param
embs = None
path_diz=sys.argv[2]
vocab = pd.read_csv(path_diz)
checkpoint_best_model=sys.argv[3]
checkpoint = torch.load(checkpoint_best_model, weights_only=False)
saved_vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
vocab_size = saved_vocab_size
model = net.ehrEncoding(vocab_size=vocab_size,
                        max_seq_len=ut.len_padded,
                        emb_size=ut.model_param['embedding_size'],
                        kernel_size=ut.model_param['kernel_size'],
                        pre_embs=embs,
                        vocab=vocab)
model.to('cuda')  #model on GPU

optimizer = torch.optim.Adam(model.parameters(),
                             lr=ut.model_param['learning_rate'],
                             weight_decay=ut.model_param['weight_decay']) 

checkpoint = torch.load(checkpoint_best_model, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval() #evaluation mode

def model_error(sequences):
    #sequence of integers
    int_sequences = [
        [int(val) for val in seq.split(',') if val.strip().isdigit()] if isinstance(seq, str) else seq 
        for seq in sequences
    ]
    
    #padding 
    len_padded = ut.len_padded  
    padded_sequences = [
        torch.cat([torch.tensor(seq, dtype=torch.long, device='cuda'), 
                   torch.zeros(len_padded - len(seq), dtype=torch.long, device='cuda')]) 
        if len(seq) < len_padded 
        else torch.tensor(seq[:len_padded], dtype=torch.long, device='cuda') 
        for seq in int_sequences
    ]
    
    #construct the trajectory
    trajectory = torch.stack(padded_sequences, dim=0)

    #ConvAE output
    with torch.no_grad():
        output = model(trajectory)
    output = output[:, 0, :]  

    #error
    mean_square_error = torch.mean((trajectory - output) ** 2, dim=1).cpu().numpy()
    mean_square_error = mean_square_error.reshape(-1, 1) 
    mean_square_error = np.concatenate([mean_square_error, 1 - mean_square_error], axis=1)
    
    return mean_square_error

explainer = LimeTextExplainer(class_names=['mean_square_error'])

#data
path_data = sys.argv[1]
cluster_df = pd.read_csv(path_data)


explanations_all = []
num_features = 32 #number of feaures to consider

for i, pat in enumerate(cluster_df['EHRseq']):  
    explanation = explainer.explain_instance(pat, model_error, num_features=num_features) #LIME explaination  
    explanations_all.append(explanation)
    

#avg weights
feature_counts = defaultdict(int)
feature_weights_sum = defaultdict(float)

for explanation in explanations_all:
    for feature, weight in explanation.as_list():
        feature_counts[feature] += 1
        feature_weights_sum[feature] += weight

feature_weights_avg = {feature: feature_weights_sum[feature] / feature_counts[feature] 
                       for feature in feature_counts}

df_weights= pd.DataFrame(list(feature_weights_avg.items()), columns=["Feature", "Average Weight"])
df_weights.to_csv("all_explanations.csv", index=False) 