import pandas as pd
import numpy as np
from IPython.display import HTML
from tqdm import tqdm
from Utils import *
import torch
from scipy.spatial.distance import cosine
from collections import OrderedDict
import pickle
from ast import literal_eval
import warnings
warnings.filterwarnings("ignore")

print("loading model...")
from transformers import AutoTokenizer, BertModel
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


#___________________________________LOAD DATASET___________________________________
# FROM ORIGINAL DATASET
"""
train_df = pd.read_excel('./Data/training.xls')

label_df_path = "./Data/all_misogyny.xlsx"
train_df = train_df.merge(pd.read_excel(label_df_path)[['meme', 'NOTmisogynous']], left_on='file_name', right_on='meme').drop_duplicates().reset_index()
train_df = get_dataset_labels(train_df)
#___________________________________COMPUTE SCORES___________________________________
train_df['sentences'] = train_df['original_text'].apply(lambda x : split_sentence(x))
train_df['sentences'] = train_df['sentences'].apply(lambda x :adjust_split(x))
train_df['sentences'] = train_df['sentences'].apply(lambda x : apply_lemmatization(x))
train_df['lemmi_text']= train_df['sentences'].apply(lambda x: ' '.join(x))

# ESCLUDO GLI ULTIMI 1000 DA USARE COME TEST
test_df = train_df[9000:]
train_df = train_df[:9000]


train_df['tokens'] = ''
context_embeddings = []
context_tokens = []

for index, row  in tqdm(train_df.iterrows()):
  tokenized_text, list_token_embeddings = text_to_emb(row.lemmi_text, tokenizer, model)
  #print(tokenized_text)
  train_df.loc[index,'tokens'] = str(tokenized_text)
  # make ordered dictionary to keep track of the position of each word
  tokens = OrderedDict()

  # loop over tokens in sensitive sentence
  for token in tokenized_text[1:-1]:
    # keep track of position of word and whether it occurs multiple times
    if token in tokens:
      tokens[token] += 1
    else:
      tokens[token] = 1

    # compute the position of the current token
    token_indices = [i for i, t in enumerate(tokenized_text) if t == token]
    current_index = token_indices[tokens[token]-1]

    # get the corresponding embedding
    token_vec = list_token_embeddings[current_index]

    # save values
    context_tokens.append(token)
    context_embeddings.append(token_vec)

# Save embeddings and tokens to a file
with open('embeddings_and_tokens.pkl', 'wb') as f:
    pickle.dump((context_embeddings, context_tokens), f)
    print("Data has been saved successfully.")

train_df.to_csv("processed_MAMI_TrainOnly.csv", sep='\t', index=False)
test_df.to_csv("processed_MAMI_TestOnly.csv", sep='\t', index=False)

"""

#LOAD PREPROCESSED
print("loading preprocessed data...")
# Load embeddings and tokens from a file
with open('embeddings_and_tokens.pkl', 'rb') as f:
    context_embeddings, context_tokens = pickle.load(f)
    print("Data has been loaded successfully.")

tags_df = pd.read_excel('./Data/training.xls')
tags_df_path = "./Data/clarifai_train.csv"
tags_df = tags_df.merge(pd.read_csv(tags_df_path), left_on='file_name', right_on='id').drop_duplicates().reset_index()
tags_df = tags_df[['file_name','Text Transcription', 'clarifai']]
tags_df['clarifai'] = tags_df['clarifai'].apply(lambda x: literal_eval(x))
tags_df['clarifai'] = tags_df['clarifai'].apply(lambda x: ['tag_'+str(element) for element in x])


train_df = pd.read_csv("processed_MAMI_TrainOnly.csv", sep='\t')
test_df = pd.read_csv("processed_MAMI_TestOnly.csv", sep='\t')

#___________________________________TOKEN SELECTION___________________________________
# Convert entire column to a list (saved as str)
train_df['tokens'] = train_df['tokens'].apply(lambda x: literal_eval(x))
valid_tokens = elements_appearing_more_than_10_times(flatten_list(train_df.tokens.values))
valid_tokens = valid_tokens + ['tag_'+str(element).lower() for element in  ['Animal', 'Broom', 'Car', 'Cartoon', 'Cat', 'Dog', 'Child', 'Crockery', 'Dishwasher', 'Kitchen', 'KitchenUtensil', 'Man', 'Woman', 'Nudity']]

# Create a new column for each unique token, with values of 1 or 0 depending on whether the token is in the "tokens" list for that row
train_df['tokens'] = train_df['tokens'].apply(lambda x: [str(element).lower() for element in x])
train_df['tokens']= train_df['tokens'].apply(lambda x: clean_tokens(x))
train_df['clarifai'] = tags_df['clarifai'][:len(train_df)] 

#train_df['tokens']= train_df.apply(lambda x: x.tokens+x.clarifai, axis=1)
train_df = tokens_to_columns(valid_tokens, train_df).copy()

#___________________________________COMPUTE SCORES___________________________________
"""
plot_scores = pd.DataFrame(columns=['token', 'Agreement', 'Hate'])

agreement_df = pd.concat([train_df.loc[train_df['soft_label_1']==1], train_df.loc[train_df['soft_label_0']==1]])
for x in valid_tokens:
  #compute p(Agreement|t)

  #if there is only one value it's 0
  if len(agreement_df[x].value_counts()) == 1 and 0 in list(agreement_df[x].values):
    p1=0
  else:
    p1 = agreement_df[x].value_counts()[1]/train_df[x].value_counts()[1]

  #compute p(Hate|t)
  #if there is only one value it's 0
  if len(train_df.loc[train_df['misogynous']==1][x].value_counts()) == 1 and 0 in list(train_df.loc[train_df['misogynous']==1][x].values):
    p2=0
  else:
    p2 = train_df.loc[train_df['misogynous']==1][x].value_counts()[1]/train_df[x].value_counts()[1]


  #plot_scores=plot_scores.append({'token':x, 'Agreement':p1, 'Hate':p2 },ignore_index=True)
  plot_scores = pd.concat([plot_scores, pd.DataFrame([{'token': x, 'Agreement': p1, 'Hate': p2}])], ignore_index=True)

plot_scores['Agreement_coordinate'] = plot_scores['Agreement'].apply(lambda x: x-(1-x))
plot_scores['Hate_coordinate'] = plot_scores['Hate'].apply(lambda x: x-(1-x))

plot_scores['occurrences'] = plot_scores['token'].apply(lambda x: train_df[x].value_counts()[1])

plot_scores.to_csv('final_scores_trainOnly.csv', sep='\t', index=False)
"""
#___________________________________LOAD SCORES___________________________________
print("loading scores...")
tags_scores = pd.read_csv('final_scores_tags_trainOnly.csv', sep='\t')
tags_scores['token']= 'tag_'+tags_scores['token']
#plot_scores= pd.concat([pd.read_csv('final_scores_trainOnly.csv', sep='\t'), tags_scores], ignore_index=True)

plot_scores= pd.read_csv('final_scores_trainOnly.csv', sep='\t')


tokens_df_10 = plot_scores[plot_scores.occurrences >= 10]
#tokens_df = plot_scores

# concateno dopo in modo da avere tutti i tag (non solo quelli con almeno 10 occorrenze)
tokens_df_10= pd.concat([tokens_df_10, tags_scores], ignore_index=True)


dev_df = train_df[8000:]

#___________________________________BASELINE NO-ESTIMATION___________________________________


#___________________________________G-MODELs TOKEN ESTIMATION___________________________________
# media pesata classica. Prima era la media dei termini moltiplicati per la distanza
print("computing context tokens...")
context_embeddings_mean = []
context_tokens_mean = []

for tk in set([str(element).lower() for element in context_tokens]):
  indexes = find_token_indices(context_tokens, tk)
  context_tokens_mean.append(tk)
  context_embeddings_mean.append(torch.mean(torch.stack(retrieve_elements(context_embeddings, indexes)) , dim=0))

print(len(context_tokens_mean))



print("computing thresholds on dev for G-Models...")
#thresholds per vicinato
best_n_somma = 0
best_n_media = 0
best_n_mediana = 0 
best_n_min = 0

#thresholds per predizione
best_t_somma = 0 
best_t_media = 0
best_t_mediana = 0 
best_t_min = 0

#performances
best_f1_somma = 0
best_f1_media = 0
best_f1_mediana = 0
best_f1_min = 0

for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    pred_somma = []
    pred_tutti_verdi = []
    pred_media = []
    pred_mediana = []
    for _, row in tqdm(dev_df.iterrows()):
        colors_agreement, _ = get_all_colors(row['tokens']+row['clarifai'], tokens_df_10)

        if 'NA' in colors_agreement:
            similar_words, distances_df, new_words  = find_similar_words(' '.join(row['tokens']), find_NA_indices(colors_agreement), tokenizer, context_tokens_mean,context_embeddings_mean, model, tokens_df_10)
            for i in range(0, len(new_words)):
                if list(distances_df.loc[(distances_df['new_token']==new_words[i])& (distances_df['distance']>=threshold)]['distance']):
                    stimated_coordinate = np.average(
                        list(distances_df.loc[(distances_df['new_token']==new_words[i]) & (distances_df['distance']>=threshold)]['Agreement_coordinate']),
                        weights=list(distances_df.loc[(distances_df['new_token']==new_words[i])& (distances_df['distance']>=threshold)]['distance']))

                    colors_agreement[find_NA_indices(colors_agreement)[0]]=stimated_coordinate
                else: #if there isn't any word above the threshold
                    colors_agreement[find_NA_indices(colors_agreement)[0]]=0
        #tolgo gli zero:
        colors_agreement = [i for i in colors_agreement if i != 0]

        if colors_agreement:

            pred_somma.append(sum(colors_agreement))
            pred_media.append(np.mean(colors_agreement))
            pred_mediana.append(np.median(colors_agreement))
            pred_tutti_verdi.append(min(colors_agreement))

        else:
            pred_somma.append(0)
            pred_media.append(0)
            pred_mediana.append(0)
            pred_tutti_verdi.append(0)

    if threshold_estimation(pred_somma, dev_df, 'disagreement')[1] > best_f1_somma:
        best_t_somma, best_f1_somma = threshold_estimation(pred_somma, dev_df, 'disagreement')
        best_n_somma = threshold

    if threshold_estimation(pred_media, dev_df, 'disagreement')[1] > best_f1_media:
        best_t_media, best_f1_media = threshold_estimation(pred_media, dev_df, 'disagreement')
        best_n_media = threshold

    if threshold_estimation(pred_mediana, dev_df, 'disagreement')[1] > best_f1_mediana:
        best_t_mediana, best_f1_mediana = threshold_estimation(pred_mediana, dev_df, 'disagreement')
        best_n_mediana = threshold

    if threshold_estimation(pred_tutti_verdi, dev_df, 'disagreement')[1] > best_f1_min:
        best_t_min, best_f1_min = threshold_estimation(pred_tutti_verdi, dev_df, 'disagreement')
        best_n_min = threshold

    print(" ___th: "+str(threshold)+"___")
    print('best_t_somma ' + str(best_t_somma))
    print('best_f1_somma ' + str(best_f1_somma))
    print('best_n_somma ' + str(best_n_somma))

    print('best_t_media ' + str(best_t_media))
    print('best_f1_media ' + str(best_f1_media))
    print('best_n_media ' + str(best_n_media))

    print('best_t_mediana ' + str(best_t_mediana))
    print('best_f1_mediana ' + str(best_f1_mediana))
    print('best_n_mediana ' + str(best_n_mediana))

    print('best_t_min ' + str(best_t_min))
    print('best_f1_min ' + str(best_f1_min))
    print('best_n_min ' + str(best_n_min))


"""
best_t_somma = 3.1
best_f1_somma = 0.5447475245646648
best_n_somma = 0.8
best_t_media = 0.2
best_f1_media = 0
best_n_media = 0.8
best_t_mediana = 0.2
best_f1_mediana = 0.5005273723248613
best_n_mediana = 0.8
best_t_min = -0.1
best_f1_min = 0.6057218415674961
best_n_min = 0.8
"""

"""
#senza zeri, th valutato tra 0.5 e 0.7
best_t_somma 3.4
best_f1_somma 0.541311553030303
best_n_somma 0.7
best_t_media 0.2
best_f1_media 0.5173944005979602
best_n_media 0.5
best_t_mediana 0.2
best_f1_mediana 0.42401905267313406
best_n_mediana 0.7
best_t_min -0.1
best_f1_min 0.6025296237728867
best_n_min 0.5
"""
"""
best_t_somma = 3.6
best_f1_somma= 0.5497666396103896
best_n_somma =0.8
best_t_media =0
best_f1_media =0
best_n_media =0
best_t_mediana= 0.2
best_f1_mediana= 0.47432315475810155
best_n_mediana =0.8
best_t_min= -0.1
best_f1_min= 0.6057218415674961
best_n_min =0.8
"""
#_______________________Performances on Test____________________
test_df['tokens'] = ''
for index, row  in tqdm(test_df.iterrows()):
  test_df.loc[index,'tokens'] = str(clean_tokens(text_to_emb(row.lemmi_text, tokenizer, model)[0]))
test_df['tokens'] = test_df['tokens'].apply(lambda x: literal_eval(x))
test_df['tokens'] = test_df['tokens'].apply(lambda x: [str(element).lower() for element in x])
train_df['tokens']= train_df['tokens'].apply(lambda x: clean_tokens(x))


test_df['clarifai'] = list(tags_df['clarifai'][len(train_df):] )
#test_df['tokens']= test_df.apply(lambda x: x.tokens+x.clarifai, axis=1)

for threshold in set([best_n_somma, best_n_media, best_n_mediana, best_n_min]):
    pred_somma = []
    pred_tutti_verdi = []
    pred_media = []
    pred_mediana = []
    for _, row in tqdm(test_df.iterrows()):
        colors_agreement, _ = get_all_colors(row['tokens']+row['clarifai'], tokens_df_10)

        if 'NA' in colors_agreement:
            similar_words, distances_df, new_words  = find_similar_words(' '.join(row['tokens']), find_NA_indices(colors_agreement), tokenizer, context_tokens_mean,context_embeddings_mean, model, tokens_df_10)
            for i in range(0, len(new_words)):
                if list(distances_df.loc[(distances_df['new_token']==new_words[i])& (distances_df['distance']>=threshold)]['distance']):
                    stimated_coordinate = np.average(
                        list(distances_df.loc[(distances_df['new_token']==new_words[i]) & (distances_df['distance']>=threshold)]['Agreement_coordinate']),
                        weights=list(distances_df.loc[(distances_df['new_token']==new_words[i])& (distances_df['distance']>=threshold)]['distance']))

                    colors_agreement[find_NA_indices(colors_agreement)[0]]=stimated_coordinate
                else: #if there isn't any word above the threshold
                    colors_agreement[find_NA_indices(colors_agreement)[0]]=0
        #tolgo gli zero:
        colors_agreement = [i for i in colors_agreement if i != 0]

        if colors_agreement:

            pred_somma.append(sum(colors_agreement))
            pred_media.append(np.mean(colors_agreement))
            pred_mediana.append(np.median(colors_agreement))
            pred_tutti_verdi.append(min(colors_agreement))

        else:
            pred_somma.append(0)
            pred_media.append(0)
            pred_mediana.append(0)
            pred_tutti_verdi.append(0)

    if threshold == best_n_somma:
      print('SOMMA')
      print(classification_report(test_df['disagreement'], [int(i>=best_t_somma) for i in pred_somma] ))
      

    if threshold == best_n_media:
      print('MEDIA')
      print(classification_report(test_df['disagreement'], [int(i>=best_t_media) for i in pred_media] ))


    if threshold == best_n_mediana:
      print('MEDIANA')
      print(classification_report(test_df['disagreement'], [int(i>=best_t_mediana) for i in pred_mediana] ))

    if threshold == best_n_min:
      print('MIN')
      print(classification_report(test_df['disagreement'], [int(i>=best_t_min) for i in pred_tutti_verdi] ))


