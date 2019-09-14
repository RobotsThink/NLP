#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-warning" align=center><b>
#     LION KING MOVIE : Reviews Sentiment Analysis</b>
# </div>

# ## Importing Required Packages

# In[1]:


# Usual packages 
import os
import json
import requests
import datetime
import time
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)
from pandas.io.json import json_normalize 
from itertools import chain
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import compress
from contractions import CONTRACTION_MAP

# text related packages
import re
import unicodedata
import emoji
import spacy
# Load the language model
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS

import gensim
import textblob
from textblob import TextBlob
from nltk.tokenize import word_tokenize


## modelling related packages
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler


# ## Scrapping Data from Web site: www.RottenTomatoes.com

# In[2]:


def web_scrapping(collect=False,no_pages=None,collection_dir=None):
    
    if(collect):
        
        # Settings required for scrapping
        headers = {
         'Referer': 'https://www.rottentomatoes.com/m/the_lion_king_2019/reviews?type=user',\
         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, likeGecko) Chrome/74.0.3729.108 Safari/537.36',\
         'X-Requested-With': 'XMLHttpRequest',\
        }


        url = 'https://www.rottentomatoes.com/napi/movie/9057c2cf-7cab-317f-876f-e50b245ca76e/reviews/user'


        payload = {
            'direction': 'next',
            'endCursor': '',
            'startCursor': ''
        }


        ### Scrap page by page as allowed

        parent_path =os.getcwd()
        ## Storing all the json data in movie_rdata dir
        reviews_path = parent_path +"\\"+collection_dir
        os.chdir(reviews_path)


        s = requests.Session()


        i=0
        while (i < no_pages):
            time.sleep(5)
            data=''
            r=''

            #print(payload,"i=",i)

            r = s.get(url, headers=headers, params=payload) # GET Call
            data = r.json()
            #print(data)

            if(data['pageInfo']['hasNextPage']):
                next_endCursor=data['pageInfo']['endCursor']

            payload = {
                'direction': 'next',
                'endCursor': next_endCursor,
                'startCursor': ''
            }

            filename="page"+str(i)+".json"
            with open(filename, 'w') as json_file:
                json.dump(data, json_file)
            i=i+1

            json_file.close()


# In[3]:


scrap_dir = "movie_rdata"
total_pages = 300 #default

scrap_choice=input("Would you like to continue with fresh web-scrapping [yes:no] ")
if(scrap_choice=='yes'):
    web_scrapping(collect=True,no_pages=total_pages,collection_dir=scrap_dir)
else:
    scrap_dir=input("Please enter the repository[Dir] from which to process the collected webscrapped pages ")


# ## Process the collected data and populate the input DF

# In[4]:


# Ask User for Movie Collection and the Directory to store

parent_path =os.getcwd()
reviews_path = parent_path +"\\"+scrap_dir
os.chdir(reviews_path)

# Process data from the collected Dir and make a df
print(reviews_path)

total_files_processed=0

for review_file in os.listdir(reviews_path):
    if review_file.endswith(".json"):
        with open(review_file) as infile:
            #print(review_file)
            jdata=json.load(infile)
            if(total_files_processed>0):
                movie_df=movie_df.append(json_normalize(jdata['reviews']))
                #print("coming here",total_files_processed)
            else:
                movie_df=json_normalize(jdata['reviews'])
                #print("first time",total_files_processed)
          
            total_files_processed=total_files_processed+1

            
print("Total files processed=",total_files_processed)
os.chdir(parent_path)


# In[5]:


# resetting index as each page indexes page as 0-9 from json collection method
movie_df=movie_df.reset_index(drop=True)


# In[6]:


movie_df.head(12)


# In[7]:


movie_df.isnull().sum()


# In[8]:


message="Review column and score column is not having null values. This is the column which we can use as per conditions. Hence, no need of impution."
print_line="----------------------------------------------------------------------------------------------------------------------------"
print(print_line)
print(message)
print(print_line)
message=""


# ### Reading Ultimate Test data on which final prediction is to be done

# In[9]:


Ultimate_TestData = pd.read_csv("test-1566619745327.csv")


# In[10]:


Ultimate_TestData.head()


# In[11]:


Ultimate_TestData.drop('ReviewID',inplace=True,axis=1)


# In[12]:


Ultimate_TestData.head()


# In[13]:


Ultimate_TestData.isnull().sum()


# In[14]:


message="We don't see any Review as null. Hence, we are all good."
print(print_line)
print(message)
print(print_line)
message=""


# ### Add the target variable for classification to input Data

# In[15]:


movie_df['targetSentiment']=[0 if x>3 else 1 for x in movie_df['score']]


# ### Dropping the columns which deemed to be ignored as per instructions

# In[16]:


movie_df.columns


# In[17]:


columns_except_review_target = ['createDate', 'displayImageUrl', 'displayName', 'hasProfanity',
       'hasSpoilers', 'isSuperReviewer', 'isVerified', 'rating','timeFromCreation', 'updateDate', 'user.accountLink',
       'user.displayName', 'user.realm', 'user.userId','score']


# In[18]:


movie_df.drop(columns_except_review_target,inplace=True,axis=1)


# In[19]:


movie_df.head()


# In[20]:


movie_df.tail()


# ## Text processing

# ### Functions for Text cleaning

# In[21]:


# Function to do the contraction operation
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)                                if contraction_mapping.get(match)                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)

    return expanded_text


# Function to do normalize operation
def normalize_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    #https://docs.python.org/2/library/unicodedata.html
    return text


# Function to add tag in sentences having did not, could not etc till punctuation
def add_splTAG_after_not(sentence=None):
    transformed = re.sub(r'\b(?:not|never)\b[\w\s]+[^\w\s]',
       lambda match: re.sub(r'(\s+)(\w+)', r'\1negtagneg\2', match.group(0)), 
       sentence,
       flags=re.IGNORECASE)
    return transformed


# Function to transform the word to its lemma form
def tokenize_lemma_clean(dataframe,column_name,dest_column_name,StopWord_Removal=False,Only_2Len_word_Removal=False):
    # text cleaning and pre-processing   
    for index,row in dataframe.iterrows():  
        row[column_name]=row[column_name].lower()

        doc=nlp(row[column_name])

        ## 1st pass of removing only 2 letter words or stop words
        if(Only_2Len_word_Removal):
            #may use STOP words for conjuction detection
            #remove words with <=2 chars
            clean_tokens1 = [token for token in doc if len(token.text)>2]
            
        if(StopWord_Removal):
        # remove stop words
            if(Only_2Len_word_Removal):
                clean_tokens1 = [token for token in clean_tokens1 if not token.is_stop]
            else:
                clean_tokens1 = [token for token in doc if not token.is_stop]
       
        ## 2nd pass of removing non-alpha words 
        if ((StopWord_Removal==True) or ( Only_2Len_word_Removal==True)):
            clean_tokens2_bool = [token.is_alpha for token in clean_tokens1]
            #clean_tokens2_bool = [token.is_alpha for token in doc]
            clean_tokens2=list(compress(clean_tokens1,clean_tokens2_bool))
        else:
            ## if only Lemma operation is needed
            clean_tokens2 = [token for token in doc]
        
        ## 3rd pass of detecting special NOT tag added earlier for did not type of sentences
        ## Where ever encounter the spl tag addtition, keep it in the review comment for later processing
        ## Hence ignore it from Lemma operation
        ## Also for the added word in spltag, keep the lemma word, so that its easier to create a dictionary 
        ## of antonymns rather than having issues with past or future tense of same words/verbs
        not_string = re.compile('negtagneg.*')
        
        clean_tokens3=[]
        for token in clean_tokens2:
            if(not_string.match(token.text)):
                neg_whole_word=""
                neg_whole_word=token.text
                word_for_lemma = neg_whole_word.split('negtagneg')
                neg_doc = nlp(word_for_lemma[1])
                neg_lemma_word=""
                for tkn in neg_doc:
                    if (len(tkn)>2): #remove 2 letter words, which will be stopwords added with negation
                        neg_lemma_word = tkn.lemma_
                
                clean_tokens3.append("negtagneg"+neg_lemma_word)
            else:
                clean_tokens3.append(token.lemma_)

        clean_text=' '.join(clean_tokens3)

        ## Add the cleaned text to the dataframe column desired
        #dataframe.at[index,column_name]=clean_text
        #print("clean text:", clean_text)
        dataframe.at[index,dest_column_name]=clean_text

        
# Function to keep specific POS tag words 
# whichever POS tag words are needed can be passed in X1 ...X4
def pos_to_keep_and_count(dataframe,column_name,dest_column_name,dest_count_coulmn_name,Take_Count=False, x1=None,x2=None,x3=None,x4=None):

    for index,row in dataframe.iterrows():
        doc=nlp(row[column_name])
        
        ## words which have different POS tags based on context / placement
        ambiguous_pos_words = ['like','love']

        # POS specific words
        pos_tokens = [token.text for token in doc if ((token.pos_ in [x1,x2,x3,x4])or(token.text in ambiguous_pos_words ))]
        

        pos_text=' '.join(pos_tokens)
        
        if(Take_Count):
            # add # of POS tags in desired column nameS
            dataframe.at[index,dest_count_coulmn_name]=len(pos_tokens)

        dataframe.at[index,dest_column_name]=pos_text

        
# Change to antonyms for words suceeding the NOT spl tags        
def antonyms_change(dataframe,column_name,dest_column_name,word=None,antonym=None):
    # text cleaning and pre-processing
    #print("coming inside antonymns")
    for index,row in dataframe.iterrows():

        doc=row[column_name].split()

        not_string = re.compile("^"+word+"$")

        reverse_tokens=[]

        for token in doc:
            if(not_string.match(token)):
                if(len(antonym)>1):
                    reverse_tokens.append(antonym)
            else:
                reverse_tokens.append(token)

        clean_text=' '.join(reverse_tokens)

        dataframe.at[index,dest_column_name]=clean_text


# In[22]:


# Dictionary list of all the negations replacement
all_antonymns_dict = {
'negtagneglike':'dislike',
'negtagneggood':'bad',
'negtagnegoriginal':'fake',
'negtagnegreal':'counterfeit',
'negtagnegfunny':'melancholy',
'negtagnegspectacular':'paltry',
'negtagnegimpress':'depress',
'negtagneginteresting':'dull',   
'negtagnegremember':'forget',
'negtagnegnear':'far',
'negtagnegfeel':'apathy',
'negtagnegadd':'miss',
'negtagnegkeep':'remove',
'negtagnegthink':'forget',
'negtagnegjustice':'injustice',
'negtagnegadd':'diminish',
'negtagnegway':'away',
'negtagnegmake':'destroy',
'negtagnegunlike':'unlike',
'negtagnegunnecessary':'unnecessary',
'negtagnegamazing': 'boring',
'negtagneganimation': 'mockery',
'negtagnegcartoon': 'spoof',
'negtagnegbelieve': 'disbelieve',
'negtagnegbetter': 'worse',
'negtagnegbig':'small',
'negtagnegcare':'disregard',
'negtagnegbad': 'bad',
'negtagnegbring':'repulse',
'negtagnegbuy':'relinquish',
'negtagneglion':'',
'negtagnegking':'',
'negtagnegall':'',
'negtagnegas':'',
'negtagnegand':'',
'negtagnega':'',
'negtagnegbe':'',
'negtagnegbut':'',
'negtagnegfor':'',
'negtagneghave':'',
'negtagnegi':'',
'negtagnegis':'',
'negtagnegin':'',
'negtagnegabove':'',
'negtagnegit':'',
'negtagnegmovie':'',
'negtagnegmuch':'',
'negtagnegnot':'',
'negtagnegof':'',
'negtagnegon':'',
'negtagnegthe':'',
'negtagnegthat':'',
'negtagnegthis':'',
'negtagnegwas':'',
'negtagnegis':'',
'negtagnegare':'',
'negtagnegwith':'',
'negtagnegof':'',
'negtagnegif':'',
'negtagnegversion':'',
'negtagnegneed':'',
'negtagnegremake':'',
'negtagnegput':'',
'negtagnegnala':'',
'negtagnegme':'',
'negtagnegbeyonce':'',
'negtagneganything':'',
'negtagnegbeing':'',
'negtagneg-PRON-':'',
'negtagnegdo':'',
'negtagneg':'',
'negtagnegsome':'',
'negtagnegstuff':'',
'negtagnegthing':'',
'negtagnegout':'',
'negtagneganimal':'',
'negtagneganimate': '',
'negtagnegfilm': '',
'negtagnegany':'',
'negnegtagneganyone':'',
'negtagnegattention':'',
'negtagnegaudience':'',
'negtagnegbeauty':'',
'negtagnegbecause':'',
'negtagnegbefore':'',
'negtagnegbegin':'',
'negtagnegthan':'',
'negtagnegbother':'',
'negtagnegcan':'',
'negtagnegdisney':'',
'negtagnegeither':'',
'negtagnegever':'',
'negtagnegnow':'',
'negtagnegalso':'',
'negtagnegagain':'',
'negtagnegthere':'',
'negtagnegthose':'',
'negtagnegvery':'',
'negtagnegwould':'',
'negtagnegwhy':'',
'negtagnegyear':'',
'negtagnegtimon':'',
'negtagnegtoo':'',
'negtagnegunder':'',
'negtagneguntil':'',
'negtagnegcharacter':'',
'negtagnegwhich':'',
'negtagnegwho':'',
'negtagnegwhere':'',
'negtagnegwhen':'',
'negtagnegabout':'',
'negtagneglet':'',
'negtagnegleast':'',
'negtagnegagain':'',
'negtagnegactor':'',
'negtagnegactually':'',
'negtagneghyena':'',
'negtagnegapart':'',


} ## TODO: will add more negative words for replacement


# substitute words 
# will be using for changing spltagged words for negatives to antonymns
def substitute_words(dataframe,source_column_name,dest_column_name,dict_rep_words=all_antonymns_dict):
    try:
        dataframe.insert(2,source_column_name,dataframe[clean_after_lemma_n_stop_2len_word])
        print ("Creating a column for the cleaned antonyms process")
    except:
        print ("Error: May be the column already exits")

    print("Negation replacement in progress ...")
        
    for key in dict_rep_words.keys():
        antonyms_change(dataframe,source_column_name,dest_column_name,key,dict_rep_words[key])
        
    print("Negation replacement all Done!")  


# ### Functions for Feature Extraction (non-lexical) from review comments

# In[23]:


# Extract Features from the review content 
# #Words
# #Capital Words
# #Question Marks 
# #Exclaimation Marks
# #NOT words
# #Emojis

class Spl_Chars_Counts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet,flags=re.IGNORECASE))

    def count_CAPS_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))
    
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x)) 
        count_love = X.apply(lambda x: self.count_regex(r'love|great|superb|amazing|awesome|beautiful|fantastic|marvellous|magnificent|stunning|spectacular|wonderful|remarkable|fabulous|incredible|astounding|astonishing|unbelievable', x))
        count_shit = X.apply(lambda x: self.count_regex(r'hate|dislik|ugly|rubbish|trash|garbage|awful|boring|slow|waste|weird|forgettable|distast|lack|shit|ass|dung|fuck|dick|cunt|bitch|bastard|ruin', x))
        count_capital_words = X.apply(lambda x: self.count_CAPS_regex(r'\b[A-Z]{2,}\b', x))
        count_quest_marks = X.apply(lambda x: self.count_regex(r'\?', x))
        count_excl_marks = X.apply(lambda x: self.count_regex(r'!', x))
        count_NOT_words = X.apply(lambda x: self.count_regex(r'\bN*oo*t*\b', x))
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
        
        df = pd.DataFrame({
                           'count_words': count_words
                           , 'count_capital_words': count_capital_words
                           , 'count_love' : count_love
                           , 'count_shit' : count_shit
                           , 'count_quest_marks': count_quest_marks
                           , 'count_excl_marks': count_excl_marks
                           , 'count_NOT_words': count_NOT_words
                           , 'count_emojis': count_emojis
                          })
        
        return df


# ### Function to check polarity of words/ phrases/ sentences

# In[24]:


def review_polarity(dataframe,column_name,dest_column_name,MULT_Factor):
    for index,row in dataframe.iterrows():
        review=TextBlob(row[column_name])

        dataframe.at[index,dest_column_name]=MULT_Factor*review.polarity


# ### Function to calculate average of w2v for list of words

# In[25]:


def compute_avg_w2v_vector(w2v_dict, review):
    list_of_word_vectors = [w2v_dict[w] for w in review if w in w2v_dict.vocab.keys()]
    
    if len(list_of_word_vectors) == 0:
        result = [0.0]*SIZE
    else:
        result = np.sum(list_of_word_vectors, axis=0) / len(list_of_word_vectors)
        
    return result


# 
# ## Starting Processing of Text 

# In[26]:


#### copying train data till this moment and safe-keeping; Using the copied variable for processing ahead 


# In[27]:


total_data = movie_df.copy(deep=True)


# In[28]:


#columns name for orginal review text
review = 'review'
review_polar = 'review_polar'

## for 1st stage of cleaning data
clean_spll_contr_accnt_NOT = 'spll_contr_accnt_NOT'

## for 2nd stage of cleaning data
clean_after_lemma_only = 'lemma_only'
clean_after_lemma_n_stop = 'lemma_stop'
clean_after_lemma_only_2len_word = 'lemma_2len_word'
clean_after_lemma_n_stop_2len_word = 'lemma_stop_2len_word'

## for 3rd stage of cleaning data
clean_after_antonymns = 'antonymns_upd'

## for POS tags
adj_count = 'adj_count'
adj_polar ='adj_polar'
adj_text='adj_text'


adv_verb_count = 'adv_verb_count'
adv_verb_polar = 'adv_verb_polar'
adv_verb_text = 'adv_verb_text'


## target column
targetSentiment = 'targetSentiment'


# In[29]:


total_data.head()


# ### 1. Spell correction

# In[30]:


#total_data[clean_spll_contr_accnt_NOT] = [TextBlob(text).correct() for text in total_data['review']]

## TextBlob Spell correction changing many words into different words, which is not desired.
## Hence not using it here


# ### 2. Expand contractions

# In[31]:


total_data[clean_spll_contr_accnt_NOT] = [expand_contractions(re.sub('’', "'", text)) for text in total_data['review']]

## Repeat the same for the final test data
Ultimate_TestData[clean_spll_contr_accnt_NOT] = [expand_contractions(re.sub('’', "'", text)) for text in Ultimate_TestData['review']]


# In[32]:


total_data.head()


# In[33]:


Ultimate_TestData.head()


# ### 3. Normalize accented characters

# In[34]:


total_data[clean_spll_contr_accnt_NOT] = [normalize_accented_chars(text) for text in total_data[clean_spll_contr_accnt_NOT]]

## Repeat the same for the final test data
Ultimate_TestData[clean_spll_contr_accnt_NOT] = [normalize_accented_chars(text) for text in Ultimate_TestData[clean_spll_contr_accnt_NOT]]


# In[35]:


total_data.head()


# ### 4. Add NOT_ after not is encountered till punctuation

# In[36]:


total_data[clean_spll_contr_accnt_NOT] = [add_splTAG_after_not(text) for text in total_data[clean_spll_contr_accnt_NOT]]

## Repeat the same for the final test data
Ultimate_TestData[clean_spll_contr_accnt_NOT] = [add_splTAG_after_not(text) for text in Ultimate_TestData[clean_spll_contr_accnt_NOT]]


# In[37]:


total_data.head()


# ### 5. Do lemma and cleaning of stop words etc

# In[38]:


tokenize_lemma_clean(total_data,clean_spll_contr_accnt_NOT,clean_after_lemma_n_stop_2len_word,StopWord_Removal=True,Only_2Len_word_Removal=False)

## Repeat the same for the final test data
tokenize_lemma_clean(Ultimate_TestData,clean_spll_contr_accnt_NOT,clean_after_lemma_n_stop_2len_word,StopWord_Removal=True,Only_2Len_word_Removal=False)


# In[39]:


total_data.head()


# ### Replace words succeeding the NOT tags with Antonyms or remove if useless

# In[40]:


## Before going ahead with identifying POS words, change the negation succeeding words to antonymns


# In[41]:


substitute_words(total_data,source_column_name=clean_after_antonymns,dest_column_name=clean_after_antonymns,dict_rep_words=all_antonymns_dict)

## Repeat the same for the final test data
substitute_words(Ultimate_TestData,source_column_name=clean_after_antonymns,dest_column_name=clean_after_antonymns,dict_rep_words=all_antonymns_dict)


# In[42]:


total_data.head()


# In[43]:


Ultimate_TestData.head()


# In[44]:


if 1==1:
    total_data.drop([clean_after_lemma_n_stop_2len_word,clean_spll_contr_accnt_NOT],axis=1,inplace=True)
    Ultimate_TestData.drop([clean_after_lemma_n_stop_2len_word,clean_spll_contr_accnt_NOT],axis=1,inplace=True)


# In[45]:


total_data.head()


# In[46]:


Ultimate_TestData.head()


# ## Feature engineering

# ### 1. Get the POS tags and counts 

# #### 1.1 Getting ADJ tags and counts

# In[47]:


#pos_to_keep_and_count(dataframe,column_name,dest_column_name,dest_count_coulmn_name,Take_Count=False,
pos_to_keep_and_count(dataframe=total_data,column_name=clean_after_antonymns,dest_column_name=adj_text,dest_count_coulmn_name=adj_count,Take_Count=True,x1='ADJ')

## Repeat the same for the final test data
pos_to_keep_and_count(dataframe=Ultimate_TestData,column_name=clean_after_antonymns,dest_column_name=adj_text,dest_count_coulmn_name=adj_count,Take_Count=True,x1='ADJ')


# In[48]:


total_data.head()


# #### 1.2 Getting ADV,VERB tags and counts

# In[49]:


pos_to_keep_and_count(dataframe=total_data,column_name=clean_after_antonymns,dest_column_name=adv_verb_text,dest_count_coulmn_name=adv_verb_count,Take_Count=True,x1='ADV',x2='VERB')

## Repeat the same for the final test data
pos_to_keep_and_count(dataframe=Ultimate_TestData,column_name=clean_after_antonymns,dest_column_name=adv_verb_text,dest_count_coulmn_name=adv_verb_count,Take_Count=True,x1='ADV',x2='VERB')


# In[50]:


total_data.head()


# ## 2. Getting polarity of review comments, adj text, adv-verb text

# In[51]:


#review_polarity(dataframe,column_name,dest_column_name,MULT_Factor)


# In[52]:


review_polarity(total_data,review,review_polar,1)

## Repeat the same for the final test data
review_polarity(Ultimate_TestData,review,review_polar,1)


# In[53]:


review_polarity(total_data,adj_text,adj_polar,1)

## Repeat the same for the final test data
review_polarity(Ultimate_TestData,adj_text,adj_polar,1)


# In[54]:


review_polarity(total_data,adv_verb_text,adv_verb_polar,1)

## Repeat the same for the final test data
review_polarity(Ultimate_TestData,adv_verb_text,adv_verb_polar,1)


# In[55]:


total_data.head()


# In[56]:


Ultimate_TestData.head()


# ## 3. Get the length, words and special chars presence from review text

# In[57]:


tc = Spl_Chars_Counts()
df_extra_feat = tc.fit_transform(total_data[review])

## Repeat the same for the final test data
tc_FinalOut = Spl_Chars_Counts()
df_extra_feat_FinalOut = tc_FinalOut.fit_transform(Ultimate_TestData[review])


# In[58]:


df_extra_feat.head()


# In[59]:


df_extra_feat_FinalOut.head()


# ## 4. Make the final data frame after feature engineering

# In[60]:


final_data = pd.concat([total_data,df_extra_feat],axis=1)

FinalOut_final_data = pd.concat([Ultimate_TestData,df_extra_feat_FinalOut],axis=1)


# In[61]:


final_data.head()


# In[62]:


FinalOut_final_data.head()


# In[63]:


### removing unecessary text data columns : adj_text,  adv_verb_text


# In[64]:


if 1==1:
    final_data.drop([adj_text,adv_verb_text,review],axis=1,inplace=True)
    FinalOut_final_data.drop([adj_text,adv_verb_text,review],axis=1,inplace=True)


# In[65]:


final_data.head()


# In[66]:


### 2 copies of data
### 1. for having non-lexical features
### 2. for having text + non-lexical features


# In[67]:


final_data_nontext = final_data.drop([clean_after_antonymns],axis=1)
FinalOut_final_data_nontext = FinalOut_final_data.drop([clean_after_antonymns],axis=1)


# In[68]:


final_data_text = final_data.copy(deep=True)
FinalOut_final_data_text = FinalOut_final_data.copy(deep=True)


# In[69]:


FinalOut_final_data_text.head()


# In[70]:


FinalOut_final_data_text.head()


# ## Data Frame to store all the results

# In[71]:


model_summary = pd.DataFrame(columns=['Model_Name','Train_Accuracy','F1_0Class','F1_1Class','Test_Accuracy','F1_0Class_test','F1_1Class_test','Features','Comments'])
model_summary_index=0


# In[72]:


model_summary


# ## ML models to be used

# In[73]:


## Function for Logistic Regression

def logistic_regr(X_train_sm,y_train_sm,X_test,C_list,Penalty,Features_name,TESTDATA=True):
    global model_summary_index
    for C in C_list:
        for penalty in Penalty:
            logisticRegr = LogisticRegression(penalty = penalty, C = C,random_state = 0)
            logisticRegr.fit(X_train_sm, y_train_sm)

            train_predicted_classes = logisticRegr.predict(X_train_sm)
            train_accuracy = accuracy_score(y_train_sm,train_predicted_classes)


            test_predicted_classes = logisticRegr.predict(X_test)

            train_accuracy = accuracy_score(y_train_sm,train_predicted_classes)
            train_f1_0 = f1_score(y_train_sm,train_predicted_classes,pos_label=0)
            train_f1_1 = f1_score(y_train_sm,train_predicted_classes,pos_label=1)
            

            
            
            print("-------------------------------------------------------------------------------------")
            print("C : ",C, "Penalty : ",penalty)
            print("TRAIN DATA ACCURACY",train_accuracy)
            print("\nTrain data f1-score for class '0'",train_f1_0)
            print("\nTrain data f1-score for class '1'",train_f1_1)
            
            model_summary.loc[model_summary_index,'Model_Name']='Logistic'
            model_summary.loc[model_summary_index,'Train_Accuracy']=train_accuracy

            model_summary.loc[model_summary_index,'F1_0Class']=train_f1_0
            model_summary.loc[model_summary_index,'F1_1Class']=train_f1_1

            if(TESTDATA==True):
                test_accuracy = accuracy_score(y_test,test_predicted_classes)
                test_f1_0 = f1_score(y_test,test_predicted_classes,pos_label=0)
                test_f1_1 = f1_score(y_test,test_predicted_classes,pos_label=1)
            
                print("TEST DATA ACCURACY",test_accuracy)
                print("\nTest data f1-score for class '0'",test_f1_0)
                print("\nTest data f1-score for class '1'",test_f1_1)

                model_summary.loc[model_summary_index,'Test_Accuracy'] = test_accuracy
                model_summary.loc[model_summary_index,'F1_0Class_test']=test_f1_0
                model_summary.loc[model_summary_index,'F1_1Class_test']=test_f1_1
                                     

            model_summary.loc[model_summary_index,'Features'] = Features_name
            model_summary.loc[model_summary_index,'Comments'] = " C : " + str(C) + "| Penalty : " + str(penalty)

            model_summary_index=model_summary_index+1
    
    #return the last one as default
    return logisticRegr

            
            

## Function for XGB model

def xgb_model(X_train_sm,y_train_sm,X_test,param_grid,Features_name,TESTDATA=True):
    global model_summary_index
    
    XGB = XGBClassifier(n_jobs=-1)
    CV_XGB = GridSearchCV(estimator=XGB, param_grid=param_grid, cv= 10)


    CV_XGB.fit(X_train_sm, y_train_sm)

    train_predictions_xgb =  CV_XGB.predict(X_train_sm)
    test_predictions_xgb =  CV_XGB.predict(X_test.values)
                                     
    train_accuracy = accuracy_score(y_train_sm,train_predictions_xgb)
    train_f1_0 = f1_score(y_train_sm,train_predictions_xgb,pos_label=0)
    train_f1_1 = f1_score(y_train_sm,train_predictions_xgb,pos_label=1)
 


    print("-------------------------------------------------------------------------------------")
    print("CV_XGB.best_score_ ",CV_XGB.best_score_)
    print("CV_XGB.best_params_ ",CV_XGB.best_params_)

    print("TRAIN DATA ACCURACY",train_accuracy)
    print("\nTrain data f1-score for class '0'",train_f1_0)
    print("\nTrain data f1-score for class '1'",train_f1_1)

                                     
    model_summary.loc[model_summary_index,'Model_Name']='XGBoost'
    model_summary.loc[model_summary_index,'Train_Accuracy']=train_accuracy

    model_summary.loc[model_summary_index,'F1_0Class']=train_f1_0
    model_summary.loc[model_summary_index,'F1_1Class']=train_f1_1
                                     
    if(TESTDATA==True):
                                     
        test_accuracy = accuracy_score(y_test,test_predictions_xgb)
        test_f1_0 = f1_score(y_test,test_predictions_xgb,pos_label=0)
        test_f1_1 = f1_score(y_test,test_predictions_xgb,pos_label=1)
        print("TEST DATA ACCURACY",test_accuracy)
        print("\nTest data f1-score for class '0'",test_f1_0)
        print("\nTest data f1-score for class '1'",test_f1_1)
                                     
        model_summary.loc[model_summary_index,'Test_Accuracy'] = test_accuracy
        model_summary.loc[model_summary_index,'F1_0Class_test']= test_f1_0
        model_summary.loc[model_summary_index,'F1_1Class_test']= test_f1_1

    model_summary.loc[model_summary_index,'Features'] = Features_name
    model_summary.loc[model_summary_index,'Comments'] = " Best_Score_ : " + str(CV_XGB.best_score_) + "| Best_Param_ : " + str(CV_XGB.best_params_)

    model_summary_index=model_summary_index+1
    
    return CV_XGB
    

## Function for RFC model
def rfc_model(X_train_sm,y_train_sm,X_test,param_grid,Features_name,TESTDATA=True):
    global model_summary_index
    
    rfc_grid = RandomForestClassifier(n_jobs=-1, max_features='sqrt')
    rfc_cv_grid = RandomizedSearchCV(estimator = rfc_grid, param_distributions = param_grid, cv = 3, n_iter=10)

    rfc_cv_grid.fit(X_train_sm, y_train_sm)

    train_predictions_rf = rfc_cv_grid.predict(X_train_sm)
    test_predictions_rf = rfc_cv_grid.predict(X_test)
                             
    train_accuracy = accuracy_score(y_train_sm,train_predictions_rf)
    train_f1_0 = f1_score(y_train_sm,train_predictions_rf,pos_label=0)
    train_f1_1 = f1_score(y_train_sm,train_predictions_rf,pos_label=1)


    print("-------------------------------------------------------------------------------------")
    print("rfc_cv_grid.best_score_ ",rfc_cv_grid.best_score_)
    print("rfc_cv_grid.best_params_ ",rfc_cv_grid.best_params_)

    print("TRAIN DATA ACCURACY",train_accuracy)
    print("\nTrain data f1-score for class '0'",train_f1_0)
    print("\nTrain data f1-score for class '1'",train_f1_1)
                             
    model_summary.loc[model_summary_index,'Model_Name']='RandomForest'
    model_summary.loc[model_summary_index,'Train_Accuracy']=train_accuracy

    model_summary.loc[model_summary_index,'F1_0Class']=train_f1_0
    model_summary.loc[model_summary_index,'F1_1Class']=train_f1_1

    if(TESTDATA==True):
        test_accuracy = accuracy_score(y_test,test_predictions_rf)
        test_f1_0 =f1_score(y_test,test_predictions_rf,pos_label=0)
        test_f1_1 =f1_score(y_test,test_predictions_rf,pos_label=1)
                             
        print("TEST DATA ACCURACY",test_accuracy)
        print("\nTest data f1-score for class '0'",test_f1_0)
        print("\nTest data f1-score for class '1'",test_f1_1)

        model_summary.loc[model_summary_index,'Test_Accuracy'] = test_accuracy
        model_summary.loc[model_summary_index,'F1_0Class_test']=test_f1_0
        model_summary.loc[model_summary_index,'F1_1Class_test']=test_f1_1

    model_summary.loc[model_summary_index,'Features'] = Features_name
    model_summary.loc[model_summary_index,'Comments'] = " Best_Score_ : " + str(rfc_cv_grid.best_score_) + "| Best_Param_ : " + str(rfc_cv_grid.best_params_)

    model_summary_index=model_summary_index+1  

    return rfc_cv_grid


# ## Model Building

# ## 1. Model building with non-text data

# In[74]:


final_data_nontext.head()


# In[75]:


X = final_data_nontext.drop([targetSentiment], axis = 1)
y = y = final_data_nontext[targetSentiment]


# #### train-test split

# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

## For future use
X_train_ntext_fut = X_train.copy(deep=True)
X_test_ntext_fut = X_test.copy(deep=True)

y_train_ntext_fut = y_train.copy(deep=True)
y_test_ntext_fut = y_test.copy(deep=True)


# #### SMOTE the train data as there is data imbalance

# In[77]:


sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)


# In[78]:


print(f"X_train.shape = {X_train.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"X_train_sm.shape = {X_train_sm.shape}")
print(f"y_train_sm.shape = {y_train_sm.shape}")
print(f"sum(y_train_sm) = {sum(y_train_sm)}")


# ### 1.1.1 Logistic Regression

# In[79]:


C_list = [0.001,0.01,0.1,1,10,100]
Penalty = ['l1','l2']

logistic_regr(X_train_sm,y_train_sm,X_test,C_list,Penalty,Features_name="Non-Text")


# ### 1.1.2 XG Boost

# In[80]:


# Use a grid over parameters of interest
param_grid = {
    'colsample_bytree': np.linspace(0.5, 0.9, 5),
    'n_estimators':[100],
    'max_depth': [10]
}

xgb_model(X_train_sm,y_train_sm,X_test,param_grid,"Non-Text")


# ### 1.1.3 RandomForest 

# In[81]:


## Use a grid over parameters of interest
## n_estimators is the number of trees in the forest
## max_depth is how deep each tree can be
## min_sample_leaf is the minimum samples required in each leaf node for the root node to split
## "A node will only be split if in each of it's leaf nodes there should be min_sample_leaf"

param_grid = {"n_estimators" : [10, 25, 50, 75, 100],
           "max_depth" : [10, 12, 14, 16, 18, 20],
           "min_samples_leaf" : [5, 10, 15, 20],
           "class_weight" : ['balanced','balanced_subsample']}


rfc_model(X_train_sm,y_train_sm,X_test,param_grid,"Non-Text")


# ## 2. Model building with Text data

# ## 2.1 Using TF-IDF only

# In[82]:


final_data_text.head()


# In[83]:


X = pd.DataFrame(final_data_text[clean_after_antonymns])
y = final_data_text[targetSentiment]


# ### train-test split

# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# ### TfIdf vectorizer

# In[85]:


# Taking both unigram and bigram
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=10000000,
                                 min_df=0.001,
                                 use_idf=True, ngram_range=(1,2))


tfidf_vectorizer.fit(X_train[clean_after_antonymns])

X_train_tfidf_matrix= tfidf_vectorizer.transform(X_train[clean_after_antonymns])
X_test_tfidf_matrix = tfidf_vectorizer.transform(X_test[clean_after_antonymns])


# In[86]:


X_train_tfidf_matrix


# In[87]:


X_test_tfidf_matrix


# In[88]:


print(tfidf_vectorizer.get_feature_names())


# #### Making tfidf dataframe

# In[89]:


X_train_tfidf_df = pd.DataFrame(X_train_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
X_test_tfidf_df  = pd.DataFrame(X_test_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())


# In[90]:


## For future use
X_train_tfidf_fut = X_train_tfidf_df.copy(deep=True)
X_test_tfidf_fut = X_test_tfidf_df.copy(deep=True)

y_train_tfidf_fut = y_train.copy(deep=True)
y_test_tfidf_fut = y_test.copy(deep=True)


# #### SMOTE the train data as there is data imbalance

# In[91]:


sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(X_train_tfidf_df, y_train)


# In[92]:


print(f"X_train_tfidf_df.shape = {X_train_tfidf_df.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"X_train_sm.shape = {X_train_sm.shape}")
print(f"y_train_sm.shape = {y_train_sm.shape}")
print(f"sum(y_train_sm) = {sum(y_train_sm)}")


# ### 2.1.1 Logistic Regression

# In[93]:


X_test=X_test_tfidf_df

C=[0.001,0.01,0.1,1,10,100]
Penalty=['l1','l2']
        
logistic_regr(X_train_sm,y_train_sm,X_test,C_list,Penalty,Features_name="TFIDF")


# ### 2.1.2 XG Boost

# In[94]:


X_test = X_test_tfidf_df

param_grid = {
    'colsample_bytree': np.linspace(0.5, 0.9, 5),
    'n_estimators':[100],
    'max_depth': [10]
}

xgb_model(X_train_sm,y_train_sm,X_test,param_grid,"TFIDF")


# ### 2.1.3 RandomForest 

# In[95]:


X_test = X_test_tfidf_df

param_grid = {"n_estimators" : [10, 25, 50, 75, 100],
           "max_depth" : [10, 12, 14, 16, 18, 20],
           "min_samples_leaf" : [5, 10, 15, 20],
           "class_weight" : ['balanced','balanced_subsample']}


rfc_model(X_train_sm,y_train_sm,X_test,param_grid,"TFIDF")


# In[96]:


model_summary.tail()


# ## 2.2 Using WordVec model for average

# In[97]:


X = pd.DataFrame(final_data_text[clean_after_antonymns])
y = final_data_text[targetSentiment]


# ### train-test split

# In[98]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[99]:


w2v_column = 'clean_wordlist'

X_train[w2v_column] = X_train[clean_after_antonymns].apply(lambda x : word_tokenize(x))
X_test[w2v_column]  = X_test[clean_after_antonymns].apply(lambda x : word_tokenize(x))


# In[100]:


SIZE = 50

model_w2v = gensim.models.Word2Vec(X_train[w2v_column], min_count=1, size=SIZE, window=5, workers=4)


# In[101]:


X_train_w2v = X_train[w2v_column].apply(lambda x: compute_avg_w2v_vector(model_w2v.wv, x))
X_test_w2v = X_test[w2v_column].apply(lambda x: compute_avg_w2v_vector(model_w2v.wv, x))


# In[102]:


X_train_w2v = pd.DataFrame(X_train_w2v.values.tolist(), index= X_train.index)
X_test_w2v = pd.DataFrame(X_test_w2v.values.tolist(), index= X_test.index)


# In[103]:


X_train_w2v.head()


# In[104]:


X_test_w2v.head()


# In[105]:


## For future use
X_train_w2v_fut = X_train_w2v.copy(deep=True)
X_test_w2v_fut = X_test_w2v.copy(deep=True)

y_train_w2v_fut = y_train.copy(deep=True)
y_test_w2v_fut = y_test.copy(deep=True)


# #### SMOTE the train data as there is data imbalance

# In[106]:


sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(X_train_w2v, y_train)


# In[107]:


print(f"X_train_w2v.shape = {X_train_w2v.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"X_train_sm.shape = {X_train_sm.shape}")
print(f"y_train_sm.shape = {y_train_sm.shape}")
print(f"sum(y_train_sm) = {sum(y_train_sm)}")


# ### 2.2.1 Logistic Regression

# In[108]:


X_test=X_test_w2v

C=[0.001,0.01,0.1,1,10,100]
Penalty=['l1','l2']
        
logistic_regr(X_train_sm,y_train_sm,X_test,C_list,Penalty,Features_name="Word2Vec")


# ### 2.2.1 XG Boost

# In[109]:


X_test = X_test_w2v

param_grid = {
    'colsample_bytree': np.linspace(0.5, 0.9, 5),
    'n_estimators':[100],
    'max_depth': [10]
}

xgb_model(X_train_sm,y_train_sm,X_test,param_grid,"Word2Vec")


# ### 2.2.1 RandomForest

# In[110]:


X_test = X_test_w2v

param_grid = {"n_estimators" : [10, 25, 50, 75, 100],
           "max_depth" : [10, 12, 14, 16, 18, 20],
           "min_samples_leaf" : [5, 10, 15, 20],
           "class_weight" : ['balanced','balanced_subsample']}


rfc_model(X_train_sm,y_train_sm,X_test,param_grid,"Word2Vec")


# ## 2.3 Using CountVectorizer model

# In[111]:


X = pd.DataFrame(final_data_text[clean_after_antonymns])
y = final_data_text[targetSentiment]


# ### train-test split

# In[112]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# ### CountVectorizer

# In[113]:


# Taking both unigram and bigram
cvect = CountVectorizer(max_df=0.9, max_features=10000000,
                                 min_df=0.001, ngram_range=(1,2))

#cvect = CountVectorizer()
cvect.fit(X_train[clean_after_antonymns])

X_train_cvect_matrix= cvect.transform(X_train[clean_after_antonymns])
X_test_cvect_matrix = cvect.transform(X_test[clean_after_antonymns])


# In[114]:


X_train_cvect_matrix


# In[115]:


print(cvect.get_feature_names())


# #### Making CountVectorize matrix as DF

# In[116]:


X_train_cvect_df = pd.DataFrame(X_train_cvect_matrix.toarray(), columns=cvect.get_feature_names())
X_test_cvect_df  = pd.DataFrame(X_test_cvect_matrix.toarray(), columns=cvect.get_feature_names())


# In[117]:


X_test_cvect_df.head()


# In[118]:


## For future use
X_train_cvect_fut = X_train_cvect_df.copy(deep=True)
X_test_cvect_fut = X_test_cvect_df.copy(deep=True)

y_train_cvect_fut = y_train.copy(deep=True)
y_test_cvect_fut = y_test.copy(deep=True)


# #### SMOTE the train data as there is data imbalance

# In[119]:


sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(X_train_cvect_df, y_train)


# In[120]:


print(f"X_train_cvect_df.shape = {X_train_cvect_df.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"X_train_sm.shape = {X_train_sm.shape}")
print(f"y_train_sm.shape = {y_train_sm.shape}")
print(f"sum(y_train_sm) = {sum(y_train_sm)}")


# ### 2.3.1 Logistic Regression

# In[121]:


X_test=X_test_cvect_df

C=[0.001,0.01,0.1,1,10,100]
Penalty=['l1','l2']
        
logistic_regr(X_train_sm,y_train_sm,X_test,C_list,Penalty,Features_name="CountVect")


# ### 2.3.1 XG Boost

# In[122]:


X_test = X_test_cvect_df

param_grid = {
    'colsample_bytree': np.linspace(0.5, 0.9, 5),
    'n_estimators':[100],
    'max_depth': [10]
}

xgb_model(X_train_sm,y_train_sm,X_test,param_grid,"CountVect")


# ### 2.3.1 RandomForest

# In[123]:


X_test = X_test_cvect_df

param_grid = {"n_estimators" : [10, 25, 50, 75, 100],
           "max_depth" : [10, 12, 14, 16, 18, 20],
           "min_samples_leaf" : [5, 10, 15, 20],
           "class_weight" : ['balanced','balanced_subsample']}


rfc_model(X_train_sm,y_train_sm,X_test,param_grid,"CountVect") 


# ## 3. Mix Text and non-Text data model

# ### 3.1 Non-text Features +  tfidf features

# In[124]:


final_data_text.head()


# In[125]:


X = final_data_text.drop(targetSentiment,axis=1)
y = final_data_text[targetSentiment]


# In[126]:


X.head()


# ### train-test split

# In[127]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[128]:


### tfidf matrix

# Taking both unigram and bigram
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=10000000,
                                 min_df=0.01,
                                 use_idf=True, ngram_range=(1,2))


tfidf_vectorizer.fit(X_train[clean_after_antonymns])

X_train_tfidf_matrix= tfidf_vectorizer.transform(X_train[clean_after_antonymns])
X_test_tfidf_matrix = tfidf_vectorizer.transform(X_test[clean_after_antonymns])


# In[129]:


X_train_tfidf_matrix


# In[130]:


X_test_tfidf_matrix


# In[131]:


print(tfidf_vectorizer.get_feature_names())


# In[132]:


X_train_tfidf_matrix


# In[133]:


X_train_tfidf_df = pd.DataFrame(X_train_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names(),index=X_train.index)
X_test_tfidf_df  = pd.DataFrame(X_test_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names(),index=X_test.index)


# In[134]:


X_train_tfidf_df.head()


# In[135]:


X_train_remaining = X_train.drop([clean_after_antonymns],axis=1)
X_test_remaining = X_test.drop([clean_after_antonymns],axis=1)


# In[136]:


X_train_remaining.head()


# In[137]:


if 1==0:
    X_train_remaining.reset_index(drop=True,inplace=True)
    X_test_remaining.reset_index(drop=True,inplace=True)

    y_train.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)


# In[138]:


final_X_train = pd.concat([X_train_remaining,X_train_tfidf_df],axis=1)
final_X_test = pd.concat([X_test_remaining,X_test_tfidf_df],axis=1)


# In[139]:


print("-------------------------------------")
print("final_X shape")
print(final_X_train.shape)
print(final_X_test.shape)

print("-------------------------------------")
print("X _remaining shape")
print(X_train_remaining.shape)
print(X_test_remaining.shape)

print("-------------------------------------")
print("X _tfidf shape")
print(X_train_tfidf_df.shape)
print(X_test_tfidf_df.shape)


# #### SMOTE the train data as there is data imbalance

# In[140]:


sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(final_X_train, y_train)


# In[141]:


print(f"final_X_train.shape = {final_X_train.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"X_train_sm.shape = {X_train_sm.shape}")
print(f"y_train_sm.shape = {y_train_sm.shape}")
print(f"sum(y_train_sm) = {sum(y_train_sm)}")


# ### 3.1.1 Logistic Regression

# In[142]:


X_test=final_X_test

C=[0.001,0.01,0.1,1,10,100]
Penalty=['l1','l2']
        
logistic_regr(X_train_sm,y_train_sm,X_test,C_list,Penalty,Features_name="Non-Text+TFIDF")


# ### 3.1.2 XG Boost

# In[143]:


X_test = final_X_test

param_grid = {
    'colsample_bytree': np.linspace(0.5, 0.9, 5),
    'n_estimators':[100],
    'max_depth': [10]
}

xgb_model(X_train_sm,y_train_sm,X_test,param_grid,"Non-Text+TFIDF")


# ### 3.1.3 RandomForest

# In[144]:


X_test = final_X_test

param_grid = {"n_estimators" : [10, 25, 50, 75, 100],
           "max_depth" : [10, 12, 14, 16, 18, 20],
           "min_samples_leaf" : [5, 10, 15, 20],
           "class_weight" : ['balanced','balanced_subsample']}


rfc_model(X_train_sm,y_train_sm,X_test,param_grid,"Non-Text+TFIDF") 


# ### 3.2 Non-text Features +  CountVectorizer features

# In[145]:


final_data_text.head()


# In[146]:


X = final_data_text.drop(targetSentiment,axis=1)
y = final_data_text[targetSentiment]


# #### train-test split

# In[147]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# #### CountVectorizer Matrix

# In[148]:


# Taking both unigram and bigram
cvect = CountVectorizer(max_df=0.9, max_features=10000000,
                                 min_df=0.001, ngram_range=(1,2))

#cvect = CountVectorizer()
cvect.fit(X_train[clean_after_antonymns])

X_train_cvect_matrix= cvect.transform(X_train[clean_after_antonymns])
X_test_cvect_matrix = cvect.transform(X_test[clean_after_antonymns])


# #### transform the FinalOut test data

# In[149]:


FinalOut_cvect_matrix = cvect.transform(FinalOut_final_data_text[clean_after_antonymns])


# #### Making CountVectorize matrix as DF

# In[150]:


X_train_cvect_df = pd.DataFrame(X_train_cvect_matrix.toarray(), columns=cvect.get_feature_names(),index=X_train.index)
X_test_cvect_df  = pd.DataFrame(X_test_cvect_matrix.toarray(), columns=cvect.get_feature_names(),index=X_test.index)


# In[151]:


X_train_cvect_df.head()


# #### Making FinalOut cvect matrix data to dataframe

# In[152]:


FinalOut_cvect_df = pd.DataFrame(FinalOut_cvect_matrix.toarray(),columns=cvect.get_feature_names())


# In[153]:


FinalOut_nontext_remaining = FinalOut_final_data_text.drop([clean_after_antonymns],axis=1)


# In[154]:


X_train_remaining = X_train.drop([clean_after_antonymns],axis=1)
X_test_remaining = X_test.drop([clean_after_antonymns],axis=1)


# In[155]:


X_train.head()


# In[156]:


final_X_train = pd.concat([X_train_remaining,X_train_cvect_df],axis=1)
final_X_test = pd.concat([X_test_remaining,X_test_cvect_df],axis=1)


# #### concat FinalOut data for non text and count vector matrix like train data above

# In[157]:


final_FinalOut_cvet_ntext = pd.concat([FinalOut_nontext_remaining,FinalOut_cvect_df],axis=1)


# In[158]:


final_FinalOut_cvet_ntext.shape


# In[159]:


print("-------------------------------------")
print("final_X shape")
print(final_X_train.shape)
print(final_X_test.shape)

print("-------------------------------------")
print("X _remaining shape")
print(X_train_remaining.shape)
print(X_test_remaining.shape)

print("-------------------------------------")
print("X _tfidf shape")
print(X_train_cvect_df.shape)
print(X_test_cvect_df.shape)


# In[160]:


sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(final_X_train, y_train)


# In[161]:


print(f"final_X_train.shape = {final_X_train.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"X_train_sm.shape = {X_train_sm.shape}")
print(f"y_train_sm.shape = {y_train_sm.shape}")
print(f"sum(y_train_sm) = {sum(y_train_sm)}")


# ### 3.2.1 Logistic Regression

# In[162]:


X_test=final_X_test

C=[0.001,0.01,0.1,1,10,100]
Penalty=['l1','l2']
        
logistic_regr(X_train_sm,y_train_sm,X_test,C_list,Penalty,Features_name="Non-Text+CountVect")


# ### 3.2.2 XG Boost

# In[163]:


X_test = final_X_test

param_grid = {
    'colsample_bytree': np.linspace(0.5, 0.9, 5),
    'n_estimators':[100],
    'max_depth': [10]
}

xgb_model(X_train_sm,y_train_sm,X_test,param_grid,"Non-Text+CountVect")


# ### 3.2.3 RandomForest

# In[164]:


X_test = final_X_test

param_grid = {"n_estimators" : [10, 25, 50, 75, 100],
           "max_depth" : [10, 12, 14, 16, 18, 20],
           "min_samples_leaf" : [5, 10, 15, 20],
           "class_weight" : ['balanced','balanced_subsample']}

print(X_test.shape)

rfc_cv_grid=rfc_model(X_train_sm,y_train_sm,X_test,param_grid,"Non-Text+CountVect")

print(X_test.shape)


# In[165]:


print(X_test.shape)


# ## After different scenarios of ML model have run, analyse the model summary for the models giving best score

# In[166]:


pd.DataFrame(model_summary).to_csv("Model_Summary_v1.csv")
print(print_line)
print(model_summary.head())
print(print_line)
print(model_summary.tail())


# ## After analyzing the xls file, using the Logistic and RFC that is the best for final prediction

# #### This model is based on 80:20 split of train data

# #### Selecting the best 3 models : 2 logistc , 1 RFC

# In[167]:


logisticRegr = LogisticRegression(penalty = 'l1', C = 1.0,random_state = 0)
logisticRegr.fit(X_train_sm, y_train_sm)

## Predicting for the given TEST data
lgr_FinalOut_ver1 = logisticRegr.predict(final_FinalOut_cvet_ntext)    
pd.DataFrame(lgr_FinalOut_ver1).to_csv("lgr_cvet_ntext_verl1_1.csv") ## Result .76 F1 score


# In[168]:


logisticRegr = LogisticRegression(penalty = 'l2', C = 1.0,random_state = 0)
logisticRegr.fit(X_train_sm, y_train_sm)

## Predicting for the given TEST data
lgr_FinalOut_ver1_1 = logisticRegr.predict(final_FinalOut_cvet_ntext) 
pd.DataFrame(lgr_FinalOut_ver1_1).to_csv("lgr_cvet_ntext_verl2_1.csv") ## Result .744 F1 score


# In[169]:


## Predicting for the given TEST data
rfc_FinalOut_ver1 = rfc_cv_grid.predict(final_FinalOut_cvet_ntext)

pd.DataFrame(rfc_FinalOut_ver1).to_csv("rfc_cvet_ntext_ver1.csv") ## Result : .72 F1 score


# ### Training on full data now

# In[170]:


## Using the best model paramaters and training on full data

full_X_train=np.concatenate([X_train_sm,X_test.values])
full_y_train=np.concatenate([y_train_sm,y_test])
#X_test=final_FinalOut_cvet_ntext

print(X_train_sm.shape)
print(X_test.shape)
print(full_X_train.shape)
print(full_y_train.shape)


# In[171]:


X_test = final_X_test
full_X_train=np.concatenate([X_train_sm,X_test.values])
full_y_train=np.concatenate([y_train_sm,y_test])
X_test=final_FinalOut_cvet_ntext

print(X_train_sm.shape)
print(X_test.shape)
print(full_X_train.shape)
print(full_y_train.shape)
## Using Best models for Predictions
## 1st :  logistic model
C_list=[1.0]
Penalty=['l1']
lgr=logistic_regr(full_X_train, full_y_train,X_test,C_list,Penalty,Features_name="FULL DATA: Non-Text+CountVect",TESTDATA=False)
    
lgr_FinalOut_ver_full2_1 = lgr.predict(final_FinalOut_cvet_ntext) ## will use this in stacking model
pd.DataFrame(lgr_FinalOut_ver_full2_1).to_csv("lgr_cvet_ntext_ver_full2_l1.csv") ## Result .7xx F1 score


lgr_full_train_v2_1_train= lgr.predict(full_X_train) ## will use this pred values for generating stacking model

## 2nd :  logistic model

C_list=[1.0]
Penalty=['l2']
lgr=logistic_regr(full_X_train, full_y_train,X_test,C_list,Penalty,Features_name="FULL DATA: Non-Text+CountVect",TESTDATA=False)

lgr_FinalOut_ver_full2_2 = lgr.predict(final_FinalOut_cvet_ntext) ## will use this in stacking model

pd.DataFrame(lgr_FinalOut_ver_full2_2).to_csv("lgr_cvet_ntext_ver_full2_l2.csv")  ## Result .7xx F1 score

lgr_full_train_v2_2_train = lgr.predict(full_X_train) ## will use this pred values for generating stacking model

## 3rd :  Random Forest model

# using the best paramaters
param_grid = {"n_estimators" : [50],
           "max_depth" : [20],
           "min_samples_leaf" : [5],
           "class_weight" : ['balanced']}


rfc_cv_grid=rfc_model(full_X_train, full_y_train,X_test,param_grid,"FULL DATA : Non-Text+CountVect",TESTDATA=False)

rfc_FinalOut_ver_full2 = rfc_cv_grid.predict(final_FinalOut_cvet_ntext)
pd.DataFrame(rfc_FinalOut_ver_full2).to_csv("rfc_cvet_ntext_ver2_.csv") ## Result .7xx F1 score


rfc_full_2_train = rfc_cv_grid.predict(full_X_train) ## will use this pred values for generating stacking model


# ### Analysis of Model Summary

# In[192]:


print('''

Top performing models on 80:20 split of train data

	Model_Name	Train_Accuracy	F1_0Class	F1_1Class	Test_Accuracy	F1_0Class_test	F1_1Class_test	Features	Comments

76	Logistic	0.94982699	0.949182243	0.950455581	0.858333333	0.901960784	0.744744745	TFIDF_CVect	 C : 1| Penalty : l1
77	Logistic	0.970011534	0.969767442	0.970251716	0.856666667	0.900232019	0.74556213	TFIDF_CVect	 C : 1| Penalty : l2

83	RandomForest	0.893310265	0.891686183	0.894886364	0.843333333	0.89044289	0.725146199	TFIDF_CVect	 Best_Score_ : 0.8457324106113033| Best_Param_ : {'n_estimators': 50, 'min_samples_leaf': 5, 'max_depth': 20, 'class_weight': 'balanced'}

''')

## These values be based on older version of models 


# ### Stacking on Full data

# In[173]:


stack_train = pd.DataFrame([rfc_full_2_train,lgr_full_train_v2_1_train,lgr_full_train_v2_2_train])


# In[174]:


stack_train = stack_train.T


# In[175]:


stack_train.head()


# In[176]:


stack_train = stack_train.as_matrix()


# In[177]:


stack_train.shape


# In[178]:


full_y_train.shape


# In[179]:


param_grid = {
     'colsample_bytree': np.linspace(0.2, 0.9, 10),
     'n_estimators':[100],
     'max_depth': [10]
}

XGB = XGBClassifier(n_jobs=-1)
stack_XGB = GridSearchCV(estimator=XGB, param_grid=param_grid, cv= 10)


# In[180]:


stack_XGB.fit(stack_train,full_y_train)


# In[181]:


stack_XGB.best_score_


# In[182]:


FinalOut_stack_train = pd.DataFrame([rfc_FinalOut_ver_full2,lgr_FinalOut_ver_full2_1,lgr_FinalOut_ver_full2_2])


# In[183]:


FinalOut_stack_train = FinalOut_stack_train.T

FinalOut_stack_train = FinalOut_stack_train.as_matrix()


# In[184]:


FinalOut_stack_train.shape


# In[185]:


stack_predict_FinalOut =  stack_XGB.predict(FinalOut_stack_train)


# In[186]:


pd.DataFrame(stack_predict_FinalOut).to_csv("stack_pred_FinalOut.csv") 


# <div class="alert alert-block alert-warning" align=center>
# </div>

# ## Plotting the Model Summary 

# In[187]:


import plotly as py
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[191]:


fig = {
    'data': [
        {'x': model_summary.index, 'y': model_summary.Train_Accuracy, 'text': model_summary.Comments,'text':model_summary.Model_Name, 'mode': 'lines', 'name': 'Train Accuracy'},
        {'x': model_summary.index, 'y': model_summary.Test_Accuracy, 'text': model_summary.Comments, 'mode': 'lines', 'name': 'Test Accuracy'},
        {'x': model_summary.index, 'y': model_summary.F1_1Class_test, 'text': model_summary.Comments, 'mode': 'lines', 'name': 'F1 Test Accuracy'},
        {'x': model_summary.index, 'y':np.zeros(87), 'text': model_summary.Features,'mode': 'markers','name': 'Model Features'}
    ],
    'layout': {
        'xaxis': {'title': 'Model #'},
        'yaxis': {'title': "Accuracy"}
    }
}
py.offline.iplot(fig)


# ## Conclusion

# #### Best model ( F1 1-Class = .76 ) was given by Logistic Regression with C=1.0 , Penalty=L1 with CountVect + Non-Text features
