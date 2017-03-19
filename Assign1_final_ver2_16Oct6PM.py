
# coding: utf-8

# In[22]:

# Question 4 - done using alpha smoothing


# In[23]:

from __future__ import division, print_function
import re
import sys
from random import random
import random
from math import log
import numpy as np
from collections import defaultdict
import collections
from numpy.random import random_sample


# In[24]:

def preprocess_line(line):
    '''this function is used for tenization of text'''
    '''Input: unpreprocess 'line' => preprocessed 'line_new' '''
    #converts the letters to lowercase
    line = line.lower() 
    line_new= "#" #adding start character
    #for loop goes through the characters of the line and checks for digits, alphabets space or period.
    #only these are retained in the new line. Since isalpha() would anything that is not an alphabet, accents and
    #other stuff should be removed.
    for chars in line:
        #incase the character is a digits, we use re.sub() to subtitue 0 in place of integer which is represented by '\d'
        if(chars.isdigit()):
            line_new += re.sub('\d', '0', chars) 
        elif(chars == " " or chars.isalpha()):
            line_new += chars
        elif(chars == "."):
            line_new += chars
        elif(chars == '\n'):
            line_new += '#' #adding end character
    return line_new


# In[25]:

###Step 1: Counting
def collect_count(line):
    '''function used to generate a trigram/bigram/unigram for the training file'''
    '''Input: preprocessed 'line' '''  
    '''Output: dictionary of trigram(tri_counts)/bigram(bi_counts)/unigram(uni_counts)'''

    #Initialization of dictionaries, we do this because we do include any n-grams as keys to our dictionaries
    tri_counts=defaultdict(int) 
    bi_counts=defaultdict(int)
    uni_counts = defaultdict(int)
    for x in char_set:
        uni_counts[x] = 0
        for y in char_set:
            bi_counts[x + y] = 0
            for z in char_set:
                tri_counts[x+y+z] = 0

    #Counting part
    for j in range(len(line)-2):
        gram = line[j:j+3]
        tri_counts[gram] += 1
    for j in range(len(line)-2): #We do this rather than len(line)-1,                                  
        gram = line[j:j+2]       #because the last bigram of the text does not give us any trigram, 
        bi_counts[gram] += 1     #hence breaks the normalization.
    for j in range(len(line)-2): #We do the same thing for unigram by the very reason above.
        gram = line[j:j+1]
        uni_counts[gram] += 1

    #Output part: 
    return(uni_counts, bi_counts, tri_counts)


# In[ ]:




# In[26]:

###Data reading and parsing    
trainingData = open('training.en', 'r')
new_line = ""
for line in trainingData:
    new_line = new_line + preprocess_line(line)

###Split the dataset into train set and cross-validation set
count = int(round(len(new_line)*0.9) + 1)
train_text = new_line[:count]
validation_text = new_line[count:]

###Do step 1(refer to the solution for detail) and get materials for smoothing
char_set = [' ','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','#','.']
vocab = len(char_set)
train_uni, train_bi, train_tri= collect_count(train_text)
dev_tri, dev_bi, dev_uni = collect_count(validation_text)


# We use alpha turing because it provides a good estimated value for values not occurring in our training data. Here the vocab size is 30 as we have 30 different characters which are unique. Hence vocab size taken as len(char_set) which is all the uniques characters in training data.

# In[27]:

###Step 3: Perplexity calculator
def perplexity(alpha,text):
    '''Input: alpha to get perplexity, text to go through'''
    '''Output: Perplexity'''
    prob_w = 0
    for i in range(2,len(text)):
        key = text[i-2:i+1]
        prob_tri = log(train_tri[key] + alpha,2) - log(train_bi[key[:2]] + (alpha*vocab),2)
        prob_w += prob_tri
    uncertainity = (-1/(len(text)-2)) * prob_w
    pp = np.power(2, (uncertainity))    
    return pp


# In[28]:

###Step 2: Find best alpha
def best_alpha_calculation():
    '''Input: list of the best-alpha candidates from near 0 to 1'''
    '''Output: minimum perplexity and its alpha of the candidates'''
    alpha_array = np.append(np.arange(1.00, 0.1, -0.1),np.arange(0.1,5e-2,-1e-3))
    min_perp = 1e10
    best_al = -1
    
    #Calculate the perplexities of the alphas in the list
    for alpha in alpha_array:
        pp = perplexity(alpha,validation_text)
        if(pp < min_perp): #Store best alpha when smaller perplexity('perp') is found
            best_al = alpha
            min_perp = pp
    print('minPP_cv =',min_perp,'at alpha =',best_al,'in the cross-validation set')
    
    #Output:
    return best_al


# In[29]:

###Initialize the probability set of our model 
model_trained = {}
for x in char_set:
    for y in char_set:
        for z in char_set:
            model_trained[x+y+z] = 0

###Step2 and 4: Find the probability set of our model and find the best alpha beforehand
def generateModelAlpha():
    '''Input: Counting of trigrams/bigrams'''
    '''Output: Probability set(or dictionary) of our model'''
    
    ###Step 2 part: Best alpha giving lowest perplexity of the cross-validation set
    model={}
    alpha = best_alpha_calculation()
    
    ###Step 4 part: Probability set via alpha-smoothing
    for keys in train_tri.keys():
        model[keys] = (train_tri[keys] + alpha)/(train_bi[keys[:2]] + (alpha*vocab))
    pp = perplexity(alpha,train_text)
    print('PP =',pp,'in the training set')
    
    ###Output:
    return model


# In[30]:

model_trained = generateModelAlpha()


# In[31]:

def test_normalization(model):
    for x in char_set:
        for y in char_set:
            p = 0
            for z in char_set:
                p += model[x+y+z]
            if (p < 0.99999999999) | (p > 1.00000001):
                print(x+y, p)


# In[32]:

test_normalization(model_trained)


# In[33]:

output = 'training_model.en'
outfile = file(output,'w')
for trigram in sorted(model_trained.keys()):
    outfile.write(trigram + "\t" + str(model_trained[trigram]) +'\n')
outfile.close()


# Question 5

# In[34]:

###Generating an output from a weighted random discrete distribution
def generate_random_sequence(distribution, N):
    ''' generate_random_sequence takes a distribution (represented as a
    dictionary of outcome-probability pairs) and a number of samples N
    and returns a list of N samples from the distribution.  
    This is a modified version of a sequence generator by fraxel on
    StackOverflow:
    http://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
    '''
    #As noted elsewhere, the ordering of keys and values accessed from
    #a dictionary is arbitrary. However we are guaranteed that keys()
    #and values() will use the *same* ordering, as long as we have not
    #modified the dictionary in between calling them.
    outcomes = np.array(distribution.keys())
    probs = np.array(distribution.values())
    #make an array with the cumulative sum of probabilities at each
    #index (ie prob. mass func)
    bins = np.cumsum(probs)
    bins = bins / bins[len(bins)-1] #Normalize 'bins' if the model probability set has 'holes' in its dictionary
    #create N random #s from 0-1
    #digitize tells us which bin they fall into.
    #return the sequence of outcomes associated with that sequence of bins
    #(we convert it from array back to list first)
    output = list(outcomes[np.digitize(random_sample(N), bins)])
    return output


# In[35]:

### Incrementally generate a sequence of characters with model probability set 
def generate_sequence(model):
    '''Input: probability set of any model '''
    '''Output: a sequence generated by taking the probability as a weight for picking characters arbitrarily'''
    items = []
    sequence = []
    temp = {}
    list_prob = {}
    holes = {}
    ### Step 1: Random generation of the first two characters with even probability weights
    ch_set = [' ','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','#','.']
    flag_valid = 0
    while flag_valid == 0:
        first_two = ''
        for i in range(2): # Generate any first two characters
            random_index_char = random.randint(0,len(ch_set)-1)
            first_two +=ch_set[random_index_char]
            sequence.append(ch_set[random_index_char])
            for third in ch_set:           
                try: # If a triplet expanding from the two characters, then the process goes to the next step
                    model[first_two+third]
                    flag_valid = 1
                except KeyError: # However if it is not, then find the third character of which the triplet is a key for the model
                    ch_set.remove(third)
                    if len(ch_set) == 0: # If there is none of such triplet, then generate the first two characters again
                        ch_set = [' ','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','#','.']
                        break
                    continue
                break
            if flag_valid == 1:
                break
    
#     #If we want to confine the first two-character history
#     first_two = '#'+'i' 
#     sequence.append('#')
#     sequence.append('i')
        
    ### Step 2: Generate characters from the third to the last one
    flag_break = 0
    error_count = 0
    ch_set = [' ','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','#','.']
    old_ch_set = [' ','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','#','.']
    while len(sequence) != 300:        
        for third in ch_set:
            tri_key = first_two + third
            try: # Case 1: If tri_key does not exist in the model probability set as a key, then find another third character
                list_prob[tri_key] = model[tri_key]
                flag_break = 0
            except KeyError:                
                ch_set.remove(third)
                if len(ch_set) == 0: # Case 2: If none of triplets of given first two characters exists, then go back to the last trigram and change the last character                    
                    old_ch_set.remove(sequence[len(sequence)-1])
                    sequence.pop()
                    first_two = sequence[len(sequence)-2] + sequence[len(sequence)-1]
                    list_prob.clear()
                    ch_set = []
                    for i in range(len(old_ch_set)):
                        ch_set.append(old_ch_set[i])
                flag_break = 1
                break            
        if flag_break == 1:
            continue
        # Case 3: If the triplet does not belong to the case 1 or 2, then generate next character by the first two
        old_ch_set = []
        for i in range(len(ch_set)):
            old_ch_set.append(ch_set[i])
        items = generate_random_sequence(list_prob, 1) #Calling the generator function gives the following character from weight-random
        char = items[0]
        first_two = char[1:]
        sequence.append(char[2])
        list_prob.clear()
    #Output: printout the sequence of 300 characters by string
    string_sequence = ''
    for i in sequence:
        string_sequence += i
    print(string_sequence)


# In[36]:

generate_sequence(model_trained)


# In[37]:

###Reading the model probability set(or dictionary)
model_file = 'model-br.en'
given_file = open(model_file,'r')
model_given = {}
for line in given_file:
    key, value = line.split('\t')
    value = value.split('\n')[0]
    model_given[key] = float(value)


# In[38]:

generate_sequence(model_given)


# In[39]:

###Intensive test unit for the 'generate_sequence' function. 
def test_generation():
    for i in range(50): #'50' in the second line means that the 'generate_sequence' is called up 50 times
        try:
            generate_sequence(model_given)
            print('')
        except (KeyError, ValueError, IndexError): #Check if there is any error occurs
            print('Error')
            break


# In[40]:

# test_generation()


# Question 6

# In[41]:

###Calculate perplexity of the model
def test_LM(model):
    '''Input: Probability set of the language model, Test text set'''
    '''Output:Perplexity'''
    infile = open('test','r')
    prob = 0
    test_data = []
    for line in infile:
        line = preprocess_line(line)
        for j in range(len(line)-3):
            trigram = line[j:j+3]
            test_data.append(trigram)
    for keys in test_data:
        prob += log(model[keys],2)
    uncertainity = (-1/len(test_data)) * prob
    perplexity = np.power(2, (uncertainity))
    return perplexity


# In[42]:

# this dictionary will hold perplexity values for different models to compare them in question 6 using 
#alpha smoothing and lamda smoothing in question7

###Call 'test_LM' function to acquire the perplexity of the test set by input model and store them as a dictionary
perp = {} # !!Turn this on/off when you want to expend the perp library or refresh
perp['en-given'] = test_LM(model_given)


# In[43]:

#this step should be repeated with different model files and the test file, to get perplexity of different models
perp_model = test_LM(model_trained)
perp[output[len(output)-2:]] = perp_model
perp


# In[44]:

#Find the language giving the lowest perplexity of the test set
def perp_min():
    minPerp = 1e100
    language = str()
    for keys in perp:
        if(perp[keys]<minPerp):
            minPerp = perp[keys]
            language = keys
    return "the language is :", language


# In[45]:

# {'en': 9.2106726118665794, 'de': 29.937418228198144, 'es': 30.08018394481353, 'given_model': 22.579108625642526}
perp_min()


# In[46]:

perp


# Question 7

# Here we have done 'lambda' interpolation. We have done lambda interpolation to compare the performance of alpha smoothing and interpolation. We expect to have a better perplexity for the test file using the model on the training data, as interpolation gives importance to the context while estimating parameters. It does this by taking into account both bigram and unigram as well as trigram

# In[47]:

### Find the best lambda1,2,3 making the lowest perplexity for the cross-validation set
def estimate_lambda(prob_uni, prob_bi, prob_tri):
    '''Input: Estimated probabilities of uni/bi/trigrams'''
    '''Output: Optimal lambdas and the minimum perplexity'''
    lambda1 = np.arange(0.9732, 0.9730, -5e-6) #Candidates for the lambda are introduces.   
    log_max_prob = -1e50
    optL1 = 0.0
    optL2 = 0.0
    optL3 = 0.0
    
    ### Throughout the layers of for loops, log(P_M{text}) is calculated in each iteration
    for x in lambda1:
        lambda2 = np.arange(0.9920-x, 0.018, -5e-5)
        for y in lambda2:
            z = 1 - (x+y)            
            log_prob_text = 0
            for i in range(2,len(validation_text)):
                key = validation_text[i-2:i+1]
                log_prob_tri = log(x*prob_tri[key] + y*prob_bi[key[1:]] + z*prob_uni[key[2]],2)                
                log_prob_text += log_prob_tri
            if log_prob_text > log_max_prob: #If probability of the text is smaller than ever,
                log_max_prob = log_prob_text #Save labmdas and the probability of the line is the               
                optL1 = x
                optL2 = y
                optL3 = z
    ### Output: with optimal lambda set, perplexity is be give as output from log(P_M{text})
    return [optL1, optL2, optL3, 2**(-1/(len(validation_text)-2)*log_prob_text)]


# In[48]:

###Q-a) Trigram character model with lambda interpolation smoothing method
def generateLamdaBasedModel():
    '''Input: Text/Counting data for the training and cross-validation set'''
    '''Output: Probability set of the model'''
    prob_uni = {}
    prob_bi = {}
    prob_tri = {}
    model_prob = {}
    
    ###Step 1: Set up P_{ML} of unigrams, bigrams, and trigrams
    for i in char_set:
        key_uni = i
        prob_uni[key_uni] = train_uni[key_uni]/(len(train_text)-2)            
        
    for i in char_set:
        zero_uni = 0
        for j in char_set:
            key_bi = i + j
            if train_uni[key_bi[0]] != 0: # Work out the conditional probability of bigrams, if possible
                prob_bi[key_bi] = train_bi[key_bi]/train_uni[key_bi[0]] 
            else: 
                prob_bi[key_bi] = 0                                                
        if zero_uni == len(char_set): #If we find a hole in bi-grams
            for j in char_set:
                key_bi = i + j
                prob_bi[key_bi] = prob_uni[j] #Fill out with uni-grams
                
    for i in char_set:
        for j in char_set:
            zero_bi = 0
            for k in char_set:
                key_tri = i + j + k
                key_first_two = key_tri[:2]
                if train_bi[key_first_two] != 0: # Work out the conditional probability of tribrams, if possible
                    prob_tri[key_tri] = train_tri[key_tri]/train_bi[key_first_two]            
                else:    
                    prob_tri[key_tri] = 0
                    zero_bi += 1
            if zero_bi == len(char_set): #If we find a hole in tri-grams
                for k in char_set:
                    key_tri = i + j + k
                    prob_tri[key_tri] = prob_uni[k] #Fill out with uni-grams
    
    ###Step 2: Find the best set of lambda minimizing the perplexity of the cross-validation text, when P_{ML} is found
    L1, L2, L3, pp = estimate_lambda(prob_uni, prob_bi, prob_tri)
    print(L1, L2, L3, pp)
    
    ###Step 3: Construct the probability set of the model with lambda interpolation method
    for i in char_set:
        for j in char_set:
            for k in char_set:
                key = i + j + k
                model_prob[key] = (L1*prob_tri[key] + L2*prob_bi[key[1:]] + L3*prob_uni[key[2]])
    return model_prob


# In[49]:

#Trigger generation of the probability set of the model - trigram characters model with lambda linear interpolation
model_lambda = generateLamdaBasedModel()


# In[50]:

#Check whether the probability set is normalized well
test_normalization(model_lambda)


# In[51]:

###Q-c) Do Q6(test the model)
perp_model_lambda = test_LM(model_lambda)
key_val = output[len(output)-2:] + '-lambda'
perp[key_val] = perp_model_lambda


# In[52]:

#See the list of models conducted the test and find the champion
print(perp_min())
print(perp)


# In[53]:

###Q-b) Do Q5(generate a weighted random sequence by language model) via lambda method
generate_sequence(model_lambda)


# Although the values of alpha smoothing and interpolation are comparable, however even with this dataset which is relatively small, interpolation provides better results. 
