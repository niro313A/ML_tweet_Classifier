import pandas as pd
import re
import numpy as np 
import io
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support as score
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
	  
	  
df = pd.read_csv(io.StringIO(uploaded['train.txt'].decode('utf-8')),sep='\t')
y = df['Label']	  


###########################			PREPROCESSING - REMOVE @USER IN TWEETS 		############################
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 
	
	
df['tidy_tweet'] = np.vectorize(remove_pattern)(df['Tweet text'], "@[\w]*")		#Remove @user
############################################################################################################



###########################			PREPROCESSING - SETMMING AND NORMALIZATION		########################
import nltk
nltk.download('stopwords'),nltk.download('porter_test')
stop_words = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


def pre_process(txt):
    z = re.sub("[^a-zA-Z]",  " ",  str(txt))
    z = re.sub(r'[^\w\d\s]', ' ', z)
    z = re.sub(r'\s+', ' ', z)
    z = re.sub(r'^\s+|\s+?$', '', z.lower())
    return ' '.join(ps.stem(term) 
        for term in z.split()
        if term not in set(stop_words)
    )
	

df['After stemming'] = df['tidy_tweet'].apply(pre_process)
print('Original Comment\n',df['tidy_tweet'].head(10),'\n\nTransformed Comment\n',df['After stemming'].head(10))
############################################################################################################




###########################			TOKENIZE 							####################################
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer =TfidfVectorizer(ngram_range=(1,2))
X_ngrams=vectorizer.fit_transform(df['After stemming'])
############################################################################################################





###########################			CLASSIFIER 							####################################
tuned_parameters = [{'kernel': ['rbf','poly'], 'gamma': [1e-3, 1e-4],	'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
					
clfSVMGs = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='roc_auc')


#Predict for test cases
uploaded =files.upload()			#Upload test data file here
test_data=pd.read_csv(io.StringIO(uploaded['test.txt'].decode('utf-8')))
VectorzedTestData = vectorizer.transform(test_data['Text'].apply(pre_process))
test_data['Label']=clfSVMGs.predict(VectorzedTestData)
test_data.to_csv('SVM_Predictions.csv', index =None, header =True)
files.download('SVM_Predictions.csv')		#Save predictions to this csv file
#############################################################################################################






#################		CALCULATE F1, PRECISSION, RECALL USING LABELED DATA		#############################
#Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf','poly'], 'gamma': [1e-3, 1e-4],	'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
					
clfSVMGs = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='roc_auc')


#Lets calculate precision, recall and f1 score by splitting the labeled data set to train and test segments 					   
#Train/Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_ngrams,y,test_size=0.2,stratify=y)					
					
clfSVMGs.fit(X_train, y_train)	
y_pred = clfSVMGs.predict(X_test)					

f1score = f1_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recallscore = recall_score(y_test, y_pred)
print('\nf1score = ',f1score)
print('\nprecision = ',precision)
print('\nrecallscore = ',recallscore)
#############################################################################################################





