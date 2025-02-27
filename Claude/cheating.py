import os
import csv
import argparse
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

LEXICON_DIRS = {
    'Hashtag': 'lexica/Hashtag-Sentiment-Lexicon',
    'Sentiment140': 'lexica/Sentiment140-Lexicon'
}


def load_datafile(filepath):
    data = []
    with open(filepath, 'r') as f:
        for row in csv.DictReader(f):
            data.append({
                'tokens': row['tweet_tokens'].split(),
                'pos_tags': row['pos_tags'].split(),
                'label': row['label']
            })
    return data
    

def load_sentiment_lexicon(lexicon):
    unigram_scores = {}
    with open(os.path.join(LEXICON_DIRS[lexicon], 'unigrams.txt'), 'r') as f:
        for line in f.readlines():
            unigram, score, _, __ = line.strip().split('\t')
            unigram_scores[unigram] = {
                'pos_score': float(score),
                'neg_score': -1 * float(score)
            }
    
    bigram_scores = {}
    with open(os.path.join(LEXICON_DIRS[lexicon], 'bigrams.txt'), 'r') as f:
        for line in f.readlines():
            bigram, score, _, __ = line.strip().split('\t')
            unigram_scores[tuple(bigram.split(' '))] = {
                'pos_score': float(score),
                'neg_score': -1 * float(score)
            }
    
    return unigram_scores, bigram_scores


# Using the specified sklearn vectorizer, trains the vectorizer on the train_set 
# and returns vectorized representations of train_set, evaluation_set
#
# Input: 
#   vectorizer - an sklearn vectorizer (Count, Tfidf) instance
#   train_set - an array of dictionaries as returned by load_datafile
#   evaluation_set - an array of dictionaries as returned by load_datafile
#
# Output:
#   A 2-tuple: (vectorized train_set representations, vectorized evaluation_set representations)
#
# hint: https://stackoverflow.com/questions/48671270/use-sklearn-tfidfvectorizer-with-already-tokenized-inputs
def extract_ngram_features(vectorizer, train_set, evaluation_set):
    # Get tokenized tweets
    train_tokens = [' '.join(item['tokens']) for item in train_set]
    eval_tokens = [' '.join(item['tokens']) for item in evaluation_set]
    
    # Fit the vectorizer on training data and transform both sets
    train_features = vectorizer.fit_transform(train_tokens)
    eval_features = vectorizer.transform(eval_tokens)
    
    return train_features, eval_features

    
# Using the specified unigram_scores and bigram_scores, 
# extracts 4 lexicon based features for specified train_set and evaluation_set
# 
# Input: 
#   unigram_scores - a dictionary as returned by load_sentiment_lexicon 
#   bigram_scores - a dictionary as returned by load_sentiment_lexicon
#   train_set - an array of dictionaries as returned by load_datafile
#   evaluation_set - an array of dictionaries as returned by load_datafile
#
# Output:
#   A 2-tuple: (vectorized train_set representations, vectorized evaluation_set representations)
#       where each representation is of dimension (# documents x 8)
#   
#       Please encode the lexicon based features in the following order:
#           0 - total count of tokens (unigrams + bigrams) in the tweet with positive score > 0
#           1 - total count of tokens (unigrams + bigrams) in the tweet with negative score > 0
#           2 - summed positive score of all the unigrams and bigrams in the tweet 
#           3 - summed negative score of all the unigrams and bigrams in the tweet
#           4 - the max positive score of all the unigrams and bigrams in the tweet
#           5 - the max negative score of all the unigrams and bigrams in the tweet
#           6 - the max of the positive scores of the last unigram / bigram in the tweet (with score > 0)
#           7 - the max of the negative scores of the last unigram / bigram in the tweet (with score > 0)
def extract_lexicon_based_features(unigram_scores, bigram_scores, train_set, evaluation_set):
    train_features = []
    eval_features = []
    
    # Process each dataset
    for dataset in [train_set, evaluation_set]:
        dataset_features = []
        
        for item in dataset:
            tokens = item['tokens']
            # Initialize feature values
            pos_count = 0
            neg_count = 0
            pos_sum = 0
            neg_sum = 0
            pos_max = 0
            neg_max = 0
            pos_last = 0
            neg_last = 0
            
            # Process unigrams
            for i, token in enumerate(tokens):
                if token in unigram_scores:
                    scores = unigram_scores[token]
                    
                    # Count tokens with positive/negative score
                    if scores['pos_score'] > 0:
                        pos_count += 1
                        pos_sum += scores['pos_score']
                        pos_max = max(pos_max, scores['pos_score'])
                        pos_last = scores['pos_score']
                    
                    if scores['neg_score'] > 0:
                        neg_count += 1
                        neg_sum += scores['neg_score']
                        neg_max = max(neg_max, scores['neg_score'])
                        neg_last = scores['neg_score']
            
            # Process bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i+1])
                if bigram in bigram_scores:
                    scores = bigram_scores[bigram]
                    
                    # Count tokens with positive/negative score
                    if scores['pos_score'] > 0:
                        pos_count += 1
                        pos_sum += scores['pos_score']
                        pos_max = max(pos_max, scores['pos_score'])
                        pos_last = scores['pos_score']
                    
                    if scores['neg_score'] > 0:
                        neg_count += 1
                        neg_sum += scores['neg_score']
                        neg_max = max(neg_max, scores['neg_score'])
                        neg_last = scores['neg_score']
            
            # Add all features
            item_features = [
                pos_count, neg_count,
                pos_sum, neg_sum,
                pos_max, neg_max,
                pos_last, neg_last
            ]
            
            dataset_features.append(item_features)
        
        # Convert to sparse matrix
        if dataset == train_set:
            train_features = csr_matrix(dataset_features)
        else:
            eval_features = csr_matrix(dataset_features)
    
    return train_features, eval_features


# Extract the 4 linguistic features for specified train_set and evaluation_set. 
#
# Input: 
#   train_set - an array of dictionaries as returned by load_datafile
#   evaluation_set - an array of dictionaries as returned by load_datafile
#
# Output:
#   A 2-tuple: (vectorized train_set representations, vectorized evaluation_set representations)
#       where each representation is of dimension (# documents x 26)
#
#       Please encode the linguistic features in the following order:
#           0 - number of tokens with all their characters capitalized
#           1-23 - separate counts of each POS tag in the following sorted order:
#               [
#                   '!', '#', '$', '&', ',', '@', 'A', 'D', 'E', 'G', 'L', 'N', 
#                   'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', '^', '~'
#               ]
#           24  - number of hashtags
#           25  - number of words with one character repeated more than two times
def extract_linguistic_features(train_set, evaluation_set):
# List of all POS tags in the specified order
    pos_tags_order = ['!', '#', '$', '&', ',', '@', 'A', 'D', 'E', 'G', 'L', 'N', 
                      'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', '^', '~']
    
    train_features = []
    eval_features = []
    
    # Process each dataset
    for dataset in [train_set, evaluation_set]:
        dataset_features = []
        
        for item in dataset:
            tokens = item['tokens']
            pos_tags = item['pos_tags']
            
            # Feature 1: All caps words
            all_caps_count = sum(1 for token in tokens if token.isupper() and len(token) > 1)
            
            # Feature 2: POS tag counts
            pos_tag_counts = {tag: 0 for tag in pos_tags_order}
            for tag in pos_tags:
                if tag in pos_tag_counts:
                    pos_tag_counts[tag] += 1
            
            # Feature 3: Number of hashtags
            hashtag_count = sum(1 for token in tokens if token.startswith('#'))
            
            # Feature 4: Elongated words (one character repeated more than twice)
            elongated_count = 0
            for token in tokens:
                if re.search(r'(.)\1{2,}', token):
                    elongated_count += 1
            
            # Combine all features
            item_features = [all_caps_count]
            item_features.extend([pos_tag_counts[tag] for tag in pos_tags_order])
            item_features.append(hashtag_count)
            item_features.append(elongated_count)
            
            dataset_features.append(item_features)
        
        # Convert to sparse matrix
        if dataset == train_set:
            train_features = csr_matrix(dataset_features)
        else:
            eval_features = csr_matrix(dataset_features)
    
    return train_features, eval_features
    

#   Extracts training and validation features as specified by the model.
#
#   Returns a 4-tuple of:
#       0 - training features (# of train documents x # of features)
#       1 - training labels (# of train documents)
#       2 - evaluation features (# of evaluation documents x # of features)
#       3 - evaluation labels (# of evaluation documents)
#
#       When encoding labels, please use the following mapping:
#           'negative' => 0
#           'neutral' => 1
#           'objective' => 2
#           'positive' => 3
def extract_features(model, lexicon, train_set, evaluation_set):
    # load sentiment lexicon
    unigram_scores, bigram_scores = load_sentiment_lexicon(lexicon)

    # Initialize with empty features
    train_features = None
    eval_features = None

    # TODO: extract n-grams 
    # -- hint: check out scikit-learn's CountVectorizer or TfidfVectorizer --
    # Extract n-grams for all models
    if 'ngram' in model.lower():
        vectorizer = CountVectorizer(
            analyzer='word',
            ngram_range=(1, 4),  # use 1 to 4-grams
            min_df=5,  # minimum document frequency
            binary=False  # use frequency counts
        )
        train_ngram_feats, eval_ngram_feats = extract_ngram_features(vectorizer, train_set, evaluation_set)
        train_features = train_ngram_feats
        eval_features = eval_ngram_feats
    
    # TODO: calculate and append lexicon based features
    # -- hint: you can use hstack and csr_matrix to append them to your n-gram features --
    if 'lex' in model.lower():
            train_lexicon_feats, eval_lexicon_feats = extract_lexicon_based_features(
                unigram_scores, bigram_scores, train_set, evaluation_set)
            if train_features is not None:
                train_features = hstack([train_features, train_lexicon_feats])
                eval_features = hstack([eval_features, eval_lexicon_feats])
            else:
                train_features = train_lexicon_feats
                eval_features = eval_lexicon_feats
    
    # TODO: calculate and append linguistic based features
    # -- hint: you can use hstack and csr_matrix to append them to your n-gram features --
    if 'ling' in model.lower():
        train_linguistic_feats, eval_linguistic_feats = extract_linguistic_features(train_set, evaluation_set)
        if train_features is not None:
            train_features = hstack([train_features, train_linguistic_feats])
            eval_features = hstack([eval_features, eval_linguistic_feats])
        else:
            train_features = train_linguistic_feats
            eval_features = eval_linguistic_feats
    
    # TODO: if the model is Custom, calculate and append any additional features you like!
    if 'custom' in model.lower():
        train_custom_feats, eval_custom_feats = extract_custom_features(train_set, evaluation_set)
        if train_features is not None:
            train_features = hstack([train_features, train_custom_feats])
            eval_features = hstack([eval_features, eval_custom_feats])
        else:
            train_features = train_custom_feats
            eval_features = eval_custom_feats
    
    # Convert labels
    label_map = {
        'negative': 0,
        'neutral': 1,
        'objective': 2,
        'positive': 3
    }
    
    train_labels = np.array([label_map[item['label']] for item in train_set])
    eval_labels = np.array([label_map[item['label']] for item in evaluation_set])
    
    return train_features, train_labels, eval_features, eval_labels
    

def train_and_evaluate(model, lexicon, train_filepath, evaluation_filepath):
    # load our dataset
    train_set = load_datafile(train_filepath)
    evaluation_set = load_datafile(evaluation_filepath)

    # extract our features
    X_train, Y_train, X_test, Y_test = extract_features(
        model, lexicon, train_set, evaluation_set
    )
    
    # train our model on our train_features (feel free to experiment with
    # other hyperparameter settings or classifiers from scikit-learn!)
    clf = SVC(kernel='linear', C=10).fit(X_train, Y_train)  
    
    # generate predictions
    Y_pred = clf.predict(X_test)
    
    # generation classification report
    classification_report = metrics.classification_report(
        Y_test, Y_pred, digits=4, labels=['negative', 'neutral', 'objective', 'positive'])
    
    return Y_pred, classification_report

def extract_custom_features(train_set, evaluation_set):
    # Initialize with empty features
    train_features = None
    eval_features = None

    train_features = []
    eval_features = []
    
    # Emoticon patterns - positive and negative
    pos_emoticon_pattern = re.compile(r'[:;=]-?[\)pPD]|[\(dD]-?[:;=]')
    neg_emoticon_pattern = re.compile(r'[:;=]-?[\(]|[\)]-?[:;=]')
    
    # Process each dataset
    for dataset in [train_set, evaluation_set]:
        dataset_features = []
        
        for item in dataset:
            tokens = item['tokens']
            tweet_text = ' '.join(tokens)
            
            # Feature 1: Punctuation intensity
            # Multiple exclamation or question marks often indicate strong sentiment
            exclamation_count = sum(1 for token in tokens if '!' in token)
            question_count = sum(1 for token in tokens if '?' in token)
            
            # Feature 2: Consecutive punctuation (e.g., !!! or ???)
            # This is often used to emphasize strong emotions
            consecutive_punct = len(re.findall(r'[!?]{2,}', tweet_text))
            
            # Feature 3: Emoticon sentiment
            # Emoticons are strong indicators of sentiment in tweets
            pos_emoticons = len(re.findall(pos_emoticon_pattern, tweet_text))
            neg_emoticons = len(re.findall(neg_emoticon_pattern, tweet_text))
            
            # Feature 4: Capitalization patterns
            # ALL CAPS words often indicate shouting or strong emotions
            all_caps_words = sum(1 for token in tokens if token.isupper() and len(token) > 1)
            
            # Feature 5: Ratio of capitalized characters
            char_count = sum(len(token) for token in tokens)
            cap_char_count = sum(sum(1 for c in token if c.isupper()) for token in tokens)
            cap_ratio = cap_char_count / max(1, char_count)  # Avoid division by zero
            
            # Feature 6: Twitter-specific features
            # Mentions (@user) can indicate conversation and often neutral content
            mention_count = sum(1 for token in tokens if token.startswith('@'))
            
            # URLs often indicate informational content, frequently neutral
            url_count = sum(1 for token in tokens if token.startswith('http'))
            
            # Feature 7: Word lengthening (e.g., "sooooo good")
            # Often used to emphasize emotions
            lengthened_words = sum(1 for token in tokens if re.search(r'(.)\1{2,}', token))
            
            # Feature 8: Intensifiers and downtoners
            # Words that modify sentiment intensity
            intensifiers = ['very', 'extremely', 'really', 'so', 'too', 'absolutely', 'completely', 'totally']
            downtoners = ['somewhat', 'slightly', 'a bit', 'kind of', 'kinda', 'rather']
            
            intensifier_count = sum(1 for token in tokens if token.lower() in intensifiers)
            downtoner_count = sum(1 for token in tokens if token.lower() in downtoners)
            
            # Feature 9: Negation words
            # Negation often flips sentiment
            negation_words = ['not', "n't", 'never', 'no', 'none', 'neither', 'nor', 'nothing']
            negation_count = sum(1 for token in tokens if token.lower() in negation_words)
            
            # Feature 10: Tweet length features
            # Very short or very long tweets might have different sentiment patterns
            token_count = len(tokens)
            avg_token_length = np.mean([len(token) for token in tokens]) if tokens else 0
            
            # Combine all features into a single vector
            item_features = [
                exclamation_count,
                question_count,
                consecutive_punct,
                pos_emoticons,
                neg_emoticons,
                all_caps_words,
                cap_ratio,
                mention_count,
                url_count,
                lengthened_words,
                intensifier_count,
                downtoner_count,
                negation_count,
                token_count,
                avg_token_length
            ]
            
            dataset_features.append(item_features)
        
        # Convert to sparse matrix
        if dataset == train_set:
            train_features = csr_matrix(dataset_features)
        else:
            eval_features = csr_matrix(dataset_features)
    
    return train_features, eval_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=['Ngram', 'Ngram+Lex', 'Ngram+Lex+Ling', 'Custom'],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon', dest='lexicon', required=True,
                        choices=['Hashtag', 'Sentiment140'])
    parser.add_argument('--train', dest='train_filepath', required=True,
                        help='Full path to the training file')
    parser.add_argument('--evaluation', dest='evaluation_filepath', required=True,
                        help='Full path to the evaluation file')
    args = parser.parse_args()

    predictions, classification_report = train_and_evaluate(
        args.model, args.lexicon, args.train_filepath, args.evaluation_filepath)
    
    print("Classification report:")
    print(classification_report)
