import string
import pandas as pd
import spacy
from sklearn.metrics import classification_report
import torch
import numpy as np
import matplotlib as mpl
from scipy.spatial.distance import cosine
from collections import Counter

# _______________________________ LOAD DATA _______________________________
def get_dataset_labels(df, columns = ['original_text','misogynous','soft_label_0','soft_label_1', 'disagreement']):
  """
  df: dataframe to elaborate
  colums: list of output columns
  ______________________________
  Extract two columns from the soft-label column to represent disagreement on the positive and negative label.
  Add a "disagreemen" column with boolean values (1 for agreement, 0 for disagreement)
  Rename the column "text" in "original text" to distiguish with the token-column "text"
  """
  df['soft_label_1']= df['NOTmisogynous'].apply(lambda x: (3-x)/3)
  df['soft_label_0']= df['NOTmisogynous'].apply(lambda x: x/3)
  df['disagreement'] = df['soft_label_0'].apply(lambda x : int(x==0 or x==1)) #1 = agreement
  df.rename({'Text Transcription': 'original_text'}, axis=1, inplace=True)
  return df[columns]

# _______________________________ PREPROCESSING _______________________________

def split_sentence(sentence):
    # Define a set of punctuation marks
    sentence = sentence.replace('\n', '.')
    punctuation = set(string.punctuation)- set(['<', '>', '@', '#',"'", '%', '*'])

    # Initialize the output list
    sentences = []

    # Split the sentence on punctuation marks
    current_sentence = ''
    for char in sentence:
        if char in punctuation:
            # Add the current sentence to the list
            if current_sentence:
                sentences.append(current_sentence.strip())
            # Start a new sentence
            current_sentence = ''
        else:
            # Add the character to the current sentence
            current_sentence += char

    # Add the final sentence to the list
    if current_sentence:
        sentences.append(current_sentence.strip())

    # Return the list of sentences
    return sentences

def adjust_split(list_sentences):
  previous_last = ''
  adjusted=[]
  unire= False
  for x in list_sentences:
    if x:
      if unire:
        unire= False
        last_split = adjusted.pop()
        last_split = last_split + ' '+x
        adjusted.append(last_split)
      else:
        # unire 'U.S.'
        last_split = x
        if x.lower()=='s' and previous_last.lower()=='u':
          last_split = adjusted.pop()
          last_split = last_split + '.'+x+'.'
          unire = True

        # unire 'U.K.'
        elif x.lower()=='k' and previous_last.lower()=='u':
          last_split = adjusted.pop()
          last_split = last_split + '.'+x+'.'
          unire = True

        # unire 'P.C.'
        elif x.lower()=='c' and previous_last.lower()=='p':
          last_split = adjusted.pop()
          last_split = last_split + '.'+x+'.'
          unire = True

        #rimuovere i numeri
        if not x.isdigit():
          adjusted.append(last_split)

      previous_last = x[len(x) - 1]
  return adjusted


punctuations = list(string.punctuation)
punctuations.remove('#')
punctuations.remove('<')
punctuations.remove('>')
punctuations.remove("'")
punctuations.append('..')
punctuations.append('...')
punctuations.append('…')
punctuations.append('–')
punctuations.append('・')

nlp = spacy.load("en_core_web_sm")

def apply_lemmatization(texts):
    """ Apply lemmatization with post tagging operations through Stanza.
    Lower case """
    noise = ['â€¢', 'a', '*', "'ll", "'", 'x', 'ã', 'â', '—', 'v', 't', '¢', '1d', 's', 't']
    processed_text = []

    for testo in texts:
        rev = []
        testo = testo.translate(str.maketrans('', '', "".join(punctuations)))

        doc = nlp(testo)

        hashtag = False
        for token in doc:
          # lemmatization only if it's not an hashtag
          if not(hashtag):
            if str(token) not in punctuations:
              rev.append((token.lemma_))
          else:
            rev.append(str(token))
          if str(token) == '#':
            hashtag = True
          else:
            hashtag = False

        # recompone hashtags, mentions, <user> and <url>
        rev = " ".join(rev).replace("# ", '#').replace("< ", '<').replace(" >", '>').replace(" @", '@')

            # Step 4: remove noise
    
        if rev not in noise:
          processed_text.append(rev)

    return processed_text

# _______________________________ EMBEDDINGS _______________________________



def bert_text_preparation(text, tokenizer):
  """
  Preprocesses text input in a way that BERT can interpret.
  """
  encoding = tokenizer(text)
  tokenized_text = encoding.tokens()
  indexed_tokens = encoding.input_ids
  segments_ids = [1]*len(indexed_tokens)

  # convert inputs to tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensor = torch.tensor([segments_ids])

  return tokenized_text, tokens_tensor, segments_tensor, encoding

def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    """
    Obtains BERT embeddings for tokens, in context of the given sentence.
    """
    # gradient calculation id disabled
    with torch.no_grad():
      # obtain hidden states
      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]

    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)

    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1,0,2)

    # intialized list to store embeddings
    token_vecs_sum = []

    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence

    # loop over tokens in sentence
    for token in token_embeddings:

        # "token" is a [12 x 768] tensor

        # sum the vectors from the last four layers
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)

    return token_vecs_sum


def text_to_emb(text, tokenizer, model):
  tokenized_text, tokens_tensor, segments_tensors, encoding = bert_text_preparation(text, tokenizer)
  list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
  recomposed_tokens, recomposed_emb = aggregate_subwords(encoding, list_token_embeddings, text)
  return recomposed_tokens, recomposed_emb



def find_similar_words(new_sent, new_token_indexes, tokenizer, context_tokens, context_embeddings, model, tokens_df):
    """
    Find similar words to the given new words in a context.

    Args:
        new_sent (str): The input sentence containing the new words.
        new_token_indexes (list): List of indexes of the new words in the sentence.

    Returns:
        tuple: A tuple containing:
            - similar_words (list): List of similar words for each new word.
            - distances_df (DataFrame): DataFrame containing the token, new_token, and distance.
            - new_words (list): List of the new words extracted from the sentence.
    """

    # embeddings for the NEW word 'record'
    list_of_distances = []
    list_of_new_embs = []
    new_words = []

    #tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(new_sent, tokenizer)
    tokenized_text, new_emb = text_to_emb(new_sent, tokenizer, model)
    #print(tokenized_text)
    if tokenized_text != new_sent.split():
      print(tokenized_text)
      print(new_sent.split())
      return [], 0, new_words

    for new_token_index in new_token_indexes:
        #new_emb = get_bert_embeddings(tokens_tensor, segments_tensors, model)[new_token_index]
        list_of_new_embs.append(new_emb[new_token_index])
        new_words.append(tokenized_text[new_token_index])

    for sentence_1, embed1 in zip(context_tokens, context_embeddings):
        for i in range(0, len(new_token_indexes)):
            cos_dist = 1 - cosine(embed1, list_of_new_embs[i])
            list_of_distances.append([sentence_1, tokenized_text[new_token_indexes[i]], cos_dist])

    distances_df = pd.DataFrame(list_of_distances, columns=['token', 'new_token', 'distance'])
    # tengo solo quelle con i token validi = che compaiono almeno 10 volte
    distances_df = distances_df.merge(tokens_df, on='token')

    similar_words = []

    for i in range(0, len(new_token_indexes)):
      if distances_df.loc[distances_df.new_token == new_sent.split(' ')[new_token_indexes[i]], 'distance'].idxmax():
        similar_words.append([distances_df.loc[distances_df.loc[distances_df.new_token == new_sent.split(' ')[new_token_indexes[i]], 'distance'].idxmax(), 'token']])
      else:
        similar_words.append([])


    return similar_words, distances_df, new_words

def aggregate_subwords(encoding, list_token_embeddings, text):
    recomposed_tokens = []  # List to store the recomposed tokens
    recomposed_emb = []  # List to store the recomposed embeddings
    hashtag = False  # Flag to indicate if a hashtag is encountered
    hashtag_emb = False  # Flag to indicate if a hashtag is part of the mean calculation

    for i in sorted(list(set(encoding.word_ids())), key=lambda x: (x is None, x)):
      #index_of_token = encoding.word_ids()[i]
      if i != None:
        #if the embedding is related to a single token
        if encoding.word_ids().count(i) ==1:
          recomposed_emb.append(list_token_embeddings[encoding.word_ids().index(i)])
        #if the embed is given by the mean of multiple tokens
        elif encoding.word_ids().count(i) >1:
          #retrive the first one
          emb = list_token_embeddings[encoding.word_ids().index(i)]
          # count the number of tokens to mean
          num = encoding.word_ids().count(i)
          # if I have to iclude an hashtag inside a mean
          if hashtag_emb:
            #remove last element (the hashag) and include it in the mean
            emb = emb + recomposed_emb.pop()
            num = encoding.word_ids().count(i)+1
            hashtag_emb = False
          for a in range(1, encoding.word_ids().count(i)):
            emb = emb + list_token_embeddings[encoding.word_ids().index(i)+a]
          emb = emb/num
          recomposed_emb.append(emb)

        start, end = encoding.word_to_chars(i)
        #print(text[start:end])
        if hashtag:
          recomposed_tokens.append('#'+text[start:end])
          hashtag=False
        elif text[start:end] == '#':
          hashtag=True
          hashtag_emb = True
          hash_emb = list_token_embeddings[encoding.word_ids().index(i)]

        else:
          #print(text[start:end])
          recomposed_tokens.append(text[start:end])
    return recomposed_tokens, recomposed_emb



def tokens_to_columns(unique_tokens, df):
  # Create a new column for each unique token in the TRAINING DATASET, with values of 1 or 0 depending on whether the token is in the "tokens" list for that row
  for token in unique_tokens:
      df[token] = df['tokens'].apply(lambda x: 1 if token in x else 0)
  return df

def elements_appearing_more_than_10_times(input_list):
    ## NB: almeno 10 volte ma potrebbero essere anche tutte e 10 nello stesso meme
    input_list = [str(element).lower() for element in input_list]
    # Step 1: Count occurrences of each element
    element_counts = Counter(input_list)

    # Step 2: Filter elements that appear more than 10 times
    filtered_elements = [element for element, count in element_counts.items() if count >= 10]

    # Step 3: Create a new list with the filtered elements
    #new_list = [element for element in input_list if element in filtered_elements]

    # Step 4: remove noise
    noise = ['â€¢', 'a', '*', "'ll", "'", 'x', 'ã', 'â', '—', 'v', 't', '¢', '1d']
    new_list = [element for element in filtered_elements if element not in noise]
    return new_list

# sistemato rispetto ad ECIR per portare in lowercase
def find_token_indices(list_to_check, token):
    """
    Finds the indices of 'NA' values in a list.

    Args:
        list_to_check (list): List to check for 'NA' values.

    Returns:
        list: List of indices where 'NA' values are found.

    """
    indices = []
    for idx, value in enumerate(list_to_check):
        if value.lower() == token:
            indices.append(idx)
    return indices

def clean_tokens(tokens):
    # Create an empty list to store the cleaned tokens
    cleaned_tokens = []

    # Initialize index
    i = 0

    while i < len(tokens):
        # Check for "'" and "t" in consecutive order
        if (tokens[i] == "'"  or tokens[i] == "'") and i + 1 < len(tokens) and tokens[i + 1] == "t":
            i += 2  # Skip the next token as well
        # Check for "'" and "s" in consecutive order
        elif (tokens[i] == "'" or tokens[i] == "'") and i + 1 < len(tokens) and tokens[i + 1] == "s":
            i += 2  # Skip the next token as well
        # Check for "*" token
        elif tokens[i] in ["*", "ç", "™"]:
            i += 1  # Skip the "*" token
        # Check for "a" token
        elif tokens[i] == "a":
            i += 1  # Skip the "a" token
        elif tokens[i] == "'":
            i += 1  # Skip the "'" token
        else:
            # If none of the conditions match, add the token to cleaned_tokens
            cleaned_tokens.append(tokens[i])
            i += 1

    return cleaned_tokens


#  _______________________________ GENERAL _______________________________
def flatten_list(sentences):
    return [item for sublist in sentences for item in sublist]

def retrieve_elements(lst, indexes):
    return [lst[i] for i in indexes]

def threshold_estimation(pred, dataset, target_col, print_bool=True):
    best_t = 0
    best_f1 = 0
    for t in np.arange(round(min(pred), 1), round(max(pred), 1), 0.1):
      t = round(t,1)
      report = classification_report(dataset[target_col], [int(i>=t) for i in pred], output_dict=True)
      if report['macro avg']['f1-score'] > best_f1:
          best_f1 = report['macro avg']['f1-score']
          best_t = t

    if round(min(pred), 1) == round(max(pred), 1): 
      for t in np.arange(0, 1, 0.1):
          t = round(t,1)
          report = classification_report(dataset[target_col], [int(i>=t) for i in pred], output_dict=True)
          if report['macro avg']['f1-score'] > best_f1:
              best_f1 = report['macro avg']['f1-score']
              best_t = t


    if print_bool:
        print('THRESHOLD: '+ str(best_t) + '\n')
        print(classification_report(dataset[target_col], [int(i>=best_t) for i in pred] ))
    return best_t, best_f1


#  _______________________________ COLOR _______________________________



def get_all_colors(tokens_list, tokens_df):
  """
  Retrieves color information for each token in the given list.
  For a given sentence create a vector with the tokens score:

  Args:
      tokens_list (list): A list of tokens (words) representing a sentence.

  Returns:
      tuple: A tuple containing two lists:
          - colors_agreement: A list of color information corresponding to the agreement coordinates
            of each token. If a token is not found in the token DataFrame, 'NA' is appended.
          - colors_hate: A list of color information corresponding to the hate coordinates
            of each token. Note: This list is currently commented out in the code.
  """
  colors_agreement = []  # List to store color information for agreement coordinates
  colors_hate = []  # List to store color information for hate coordinates

  for token in tokens_list:
      if token.lower() not in list(tokens_df['token']):
          colors_agreement.append('NA')  # Token not found in token DataFrame, 'NA' is appended
      else:
          # Token found in token DataFrame, append its agreement coordinate color information
          colors_agreement.append(tokens_df.loc[tokens_df['token'] == token.lower(),
                                                  'Agreement_coordinate'].values[0])

      # Note: The following line is currently commented out in the code
      # colors_hate.append(tokens_df.loc[tokens_df['token'] == token.lower(), 'Hate_coordinate'].values[0])

  return colors_agreement, colors_hate

def find_NA_indices(list_to_check):
    """
    Finds the indices of 'NA' values in a list.

    Args:
        list_to_check (list): List to check for 'NA' values.

    Returns:
        list: List of indices where 'NA' values are found.

    """
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == 'NA':
            indices.append(idx)
    return indices



def colorize(attrs, cmap='PiYG'):
    """
    Colorizes a list of attributes using a specified colormap.

    Args:
        attrs (list): List of attributes.
        cmap (str, optional): Colormap name. Defaults to 'PiYG'.

    Returns:
        list: List of colors in hexadecimal format.

    """

    indexes = []

    if 'NA' in attrs:
        # Find indices of 'NA' values in attrs
        indexes = find_NA_indices(attrs)

        # Replace 'NA' values with 0
        for i in indexes:
            attrs[i] = 0

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmap = mpl.colormaps.get_cmap(cmap)#.cm.get_cmap(cmap)

    # Convert attribute values to colors using the colormap
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))

    if indexes:
        # Set colors for 'NA' values to gray (#A9A9A9)
        for i in indexes:
            colors[i] = '#A9A9A9'  # '#FFFF00'

    return colors