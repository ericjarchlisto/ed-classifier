import json
import logging
import math
import re
#import resource # Linux specific
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
# from elasticsearch_dsl.search import Search

from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# params
CONFIDENCE_THRESHOLD = 0.6          # minimum `decision_function` value for a datapoint's class prediction
EXTRACTION_CHUNK_SIZE = 5000
PROCESSING_CHUNK_SIZE = 1000
BETA = 0.25  # Prioritize precision over recall
MAX_CLASSES = 500
DATA_LEN = 26040
N_TRAIN = int(DATA_LEN*0.75)

DATASET_PATH = './data/data-sample-invoices.csv'


@dataclass
class InvoiceRecordForClassification:
    counterparty_name: str
    counterparty_rfc: str
    descriptions: str
    # no 'text', no 'prepayment as inputs anymore

@dataclass
class InvoiceRecord(InvoiceRecordForClassification):
    id: int
    values: Dict[str, Any]  # to know wich fields are populated


class PartSelector(BaseEstimator, TransformerMixin):
    """Clean and transform text from each document for Vectorizing
        implements fit() and transform() according to sklearn API conventions
    """
    def __init__(self, part):
        self.part = part

    def fit(self, x, y=None):
        # no fitting required
        return self

    def _convertAccented(self, text, pattobj):
            '''
            Restore characters from lowercase text, like "&oacute;" into "ó"
            '''
            accented = {
                'a':'á',
                'e':'é',
                'i':'í',
                'o': 'ó',
                'u':'ú'
            }
            
            def accentRepl(matchobj):
                letter = matchobj.group(1)
                return accented[letter]
            
            return pattobj.sub(accentRepl, text)

    def transform(self, invoices: List[InvoiceRecordForClassification]):
        # For every invoice's text features convert into vectorizable plain text
        # This is the 1st step in the vectorizer pipeline. 2nd is Hashingvectorizer
        patt = r'&([aeiou])acute;'  # vowel is captured by group 1
        accent_rgx = re.compile(patt)      # compiled beforehand for performance

        def f(i: InvoiceRecordForClassification) -> str:
            if self.part == 'lineitems':
                # Part of the workflow could be when lineitems are being filled
                return i.descriptions.replace('\n', ' ')
            else:
                # Input features: counterparty_name, counterparty_rfc, descriptions
                all_text = ' '.join([
                    i.counterparty_name or '',
                    i.counterparty_rfc or ''
                ]).replace('\n', ' ') + (i.descriptions or '').replace('\n', ' ')
                
                # should we lowercase it all?
                #all_text = self._convertAccented(all_text.lower(), accent_rgx)

                if self.part.startswith('extract_regex:'):
                    # Customer workflow may require to predict based on specific words from text
                    matches = re.findall(self.part[len('extract_regex:'):], all_text, re.IGNORECASE)
                    if matches:
                        # print 'MATCHES', ' '.join(matches)
                        return ' '.join(matches)
                    else:
                        # No matches
                        return 'NA'
                else:
                    return all_text

        return map(f, invoices)


class ExtraDataClassifierSimple(object):
    '''
    Customer-agnostic classifier, don't care about workflows or parts; 
    we only receive the data and predict for two fields: nature and cost center.
    we do not use 'text' as input anymore.
    we do not use timestamp from invoices

    A classifier instance is created on a per-workflow basis (?)
    '''

    clfs: Dict[str, SGDClassifier]  # field name -> sklearn SVM Classifier
    vectorizers: Dict[str, Pipeline]  # field name -> Transformer-Vectorizer object
    targets: Dict[str, List[str]]   # field name -> [label values]
    last_modified_on: int  # UTC timestamp, for selecting the data to use for partial fit

    def __init__(self, clf_factory=None, part='all', updateOnStart=False, vect='hash'):
            # self.logger = logging.getLogger(f'{__package__}.{__name__}.simple-classif')
            self.df = pd.read_csv(DATASET_PATH, index_col=0)
            self.prepareDataFrame()
            self.vect=vect
            self.fields = ['nature', 'cost_center']  # targets
            self.targets = {field: [] for field in self.fields}  # field name -> labels values array
            self.vectorizers = {field: None for field in self.fields}  # field name -> sklearn vectorizer v(invoices_with_field) 
            self.data_vectorized = {field: None for field in self.fields}  # field name -> output of vectorizers[field]
            self.last_modified_on = 0
            self.part = part  
            self.is_viable = False   # huh?
            self.clfs = {}  # field name -> sklearn classifier: f(invoices, target_labels_array)
            if updateOnStart:
                print("Vectorize and train classifiers...")
                self.update() # the very first train
            else:
                print("only vectorize Data...")
                self.onlyVectorizeData()

    def prepareDataFrame(self):
        '''Standardize dataframe columns for Data extraction;
            cast nature and cost_center into category (int) types'''
        if 'nature' in self.df:
            self.df['nature'] = self.df['nature'].astype('Int64')
            self.df['nature'] = self.df['nature'].astype('category')            

        else: 
            # self.logger.warning("INIT \tno `nature` in dataframe")
            print("INIT \tno `nature` in dataframe")

        if 'cost_center' in self.df:
            self.df['cost_center'] = self.df['cost_center'].astype('Int64')
            self.df['cost_center'] = self.df['cost_center'].astype('category')
        else: 
            # self.logger.warning("INIT \tno `cost_center` in dataframe")
            print("INIT \tno `cost_center` in dataframe")
        
        if 'text' in self.df:
            # remove the overall null column of `text`
            self.df.drop('text', inplace=True, axis=1)
        
        # self.logger.info("Data columns preparation finished")
        print("Data columns preparation finished")


    def create_svm_classifier(self):
        '''Create a Stochastic Gradient Descent SVM'''
        # no customization YET
        return SGDClassifier(alpha=1e-5, n_jobs=-1) # by default is an SVM

    def create_log_classifier(self):
        '''Create a Stochastic Gradient Descent Logistic Regressor'''
        # no customization YET
        return SGDClassifier(alpha=1e-5, loss='log', n_jobs=-1) 

    def create_perceptron_classifier(self):
        '''Create a Stochastic Gradient Descent Perceptron Classifier'''
        # no customization YET
        return SGDClassifier(alpha=1e-5, loss='perceptron', n_jobs=-1) 
    
    

    def get_classes(self, field:str) -> Set[str]:
        '''Return set of unique values for the specified column in df'''
        if field in self.df:
            return set(self.df[field].unique())

        return set()

    def _process_hit(self, r: Any) -> Optional[InvoiceRecord]:
        '''Receive a dataframe row, return an InvoiceRecord'''
        # If we used timestamp, we would update self.last_modified here
        def _get_value(r: Dict[str, Any], field: str) -> Optional[Any]:
            '''Return specified value from row if not empty'''

            value = r[field]

            if isinstance(value, str) and value.strip() != '' or isinstance(value,int):
                return value
            # Values could be NaN objects (stored as float)
            return None

        values = {}  # dictionary of target field:value
        
        for field in self.fields: # target fields
            value = _get_value(r, field)
            if value is not None:
                values[field] = value
        
        # descriptions may be NaN object. If so, convert to empty str
        descriptions = _get_value(r, 'descriptions') or ''
        # return InvoiceRecord if there is at least 1 target field
        if values:  
            return InvoiceRecord(
                id=r['id'],
                values=values,  # target values of this invoice
                descriptions=descriptions,
                counterparty_name=r['counterparty_name'],
                counterparty_rfc=r['counterparty_rfc'],
            )

        else:
            # this invoice is not useful for training
            return None

    def _add_descriptions(self, chunk: Dict[int, InvoiceRecord]) -> None:
            # In production we would query the descriptions from associated LineItem objects
            # But our data already holds this information in values['descriptions] if it was not NaN
            pass
           
    def _vectorize(self, invoices: List[InvoiceRecord]) -> None:
            '''
            Called by _process_chunk
            Receives a processing chunk and for every field, creates a 
            binary count sparse matrix using the invoices with the target.
            '''
            # Predictor is an SVM trained on Hashing counts sparse matrix.
 
            for field in self.fields:
                if not self.vectorizers[field]:  # don't think this clause is ever true
                    print("No field in vectorizer")
                    continue
                
                # Grab all invoices that have target value 
                invoices_with_field = [i for i in invoices if field in i.values]
                # self.logger.info(f'Processing field {field}: {len(invoices_with_field)} invoices')

                # Only store vectorized representation, otherwise memory usage grows very rapidly
                if self.vect=='hash':
                    vectorized = self.vectorizers[field].transform(invoices_with_field)  # triggers both clean and vectorize steps on IR attribute
                else:
                    vectorized = self.vectorizers[field].fit_transform(invoices_with_field)  # triggers both clean and vectorize steps on IR attribute


                if self.data_vectorized[field] is not None:
                    # append it to the already stored sparse matrix
                    self.data_vectorized[field] = sparse.vstack([self.data_vectorized[field], vectorized])
                else:
                    # first time adding data for this field
                    self.data_vectorized[field] = vectorized

                self.targets[field] += [json.dumps(i.values[field]) for i in invoices_with_field]  # y values for this field's predictor

            ## self.logger.info('%.1fMB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

    def extract_data(self) -> None:
        '''
        Processes invoices into records, calls vectorizing function on every chunk of records 
        which updates self.data_vectorized and self.targets.
        It 
        '''
        def _process_chunk(c: Dict[int, InvoiceRecord]) -> None:
            # self._add_descriptions(c)  # invoices already have not-null descriptions
            self._vectorize(list(c.values())) # vectorize invoices list
            c.clear()

        data = self.df[['id', 'counterparty_name', 'counterparty_rfc', 'descriptions', 'nature', 'cost_center']]
        # self.logger.info('Extracting data from %s new invoices', len(data))
        print(f'Extracting data from {len(data)} new invoices')

        if self.vect=='hash':
            chunk_size=EXTRACTION_CHUNK_SIZE
        else:
            chunk_size = len(self.df)

        chunk: Dict[int, InvoiceRecord] = {}
        for index, hit in tqdm(data.iterrows(), total=len(data)):
            # iterate over the rows of data, named hit
            record = self._process_hit(hit)  # returns InvoiceRecord
            if record is not None:
                chunk[hit['id']] = record # can it be index?

            if len(chunk) >= chunk_size:

                _process_chunk(chunk)  # vectorize invoices so far and then clear the chunk
        # Leftover results (modulus PROCESSING_CHUNK_SIZE)
        if chunk:
            _process_chunk(chunk)

    def onlyVectorizeData(self):
        '''Only populates vectorizer and data_vectorized dicts, 
        avoids training and populating the clfs dict'''
        if not self.fields:
                print("no fields to predict")
                return

        # 1. Determine target viability of a field
        print("1) Getting target classes viability")
        for field in self.fields:
            classes = self.get_classes(field)
            # self.logger.info('[%s] Extracted %s unique classes', field, len(classes))
            print(f'[{field}] Extracted {len(classes)} unique classes')
            # self.logger.info('[%s] %s', field, repr(classes))
            
            if 2 <= len(classes) <= MAX_CLASSES:
                self.is_viable = True  # no transcendence
                part = 'all' # otherwise comes from worklow.fields.fieldname.classifier.part
                
                if self.vect=='hash':
                    self.vectorizers[field] = Pipeline([
                        ('selector', PartSelector(part)),
                        ('vectorizer',
                        HashingVectorizer(n_features=262144, ngram_range=(1, 2),
                                        binary=True, strip_accents='ascii'))
                    ])
                # HashVectorizer uses occurrence counts (0 or 1), and n_features = 2**18. could we tweak up to 2**20 which is default?
                
                # STATEFUL VECTORIZERS - use in-memory vocabulary, do not support incremental learning. Only meant for benchmarking keywords
                if self.vect=='ngram':
                    self.vectorizers[field] = Pipeline([
                        ('selector', PartSelector(part)),
                        ('vectorizer',
                        CountVectorizer(min_df=5, ngram_range=(1,2)))
                    ])

                if self.vect=='tfidf':
                    self.vectorizers[field] = Pipeline([
                        ('selector', PartSelector(part)),
                        ('vectorizer',
                        TfidfVectorizer(min_df=5))
                    ])

            else:
                # self.logger.info('[%s] Not viable: %s classes', field, len(classes))
                print(f"[{field}] Not viable: {len(classes)} classes")

        # 2. Extract vectorized data for the viable fields, register time taken
        print("2) Extracting and vectorizing...")
        start_time = time.time()
        self.extract_data()
        # self.logger.info('Extracted data in %s seconds', time.time() - start_time)
        print(f"Extracted data in {time.time() - start_time} seconds")


    def update(self):
            """Process next batch of invoices (based on categorization time),
            re-train the classifier using the whole set
            how often is this method called?

            All fields in self.fields are predicted on the same input features defined in InvoiceRecord
            """

            if not self.fields:
                print("no fields to predict")
                return

            # 1. Determine target viability of a field
            print("1) Getting target classes viability")
            for field in self.fields:
                classes = self.get_classes(field)
                # self.logger.info('[%s] Extracted %s unique classes', field, len(classes))
                print(f'[{field}] Extracted {len(classes)} unique classes')
                # self.logger.info('[%s] %s', field, repr(classes))
                
                if 2 <= len(classes) <= MAX_CLASSES:
                    self.is_viable = True  # no transcendence
                    part = 'all' # otherwise comes from worklow.fields.fieldname.classifier.part
                    
                    self.vectorizers[field] = Pipeline([
                        ('selector', PartSelector(part)),
                        ('vectorizer',
                        HashingVectorizer(n_features=262144, ngram_range=(1, 2),
                                        binary=True, strip_accents='ascii'))
                    ])
                    # HashVectorizer uses occurrence counts (0 or 1), and n_features = 2**18. could we tweak up to 2**20 which is default?
                else:
                    # self.logger.info('[%s] Not viable: %s classes', field, len(classes))
                    print(f"[{field}] Not viable: {len(classes)} classes")

            # 2. Extract vectorized data for the viable fields, register time taken
            print("2) Extracting and vectorizing...")
            start_time = time.time()
            self.extract_data()
            # self.logger.info('Extracted data in %s seconds', time.time() - start_time)
            print(f"Extracted data in {time.time() - start_time} seconds")
            
            # 3. Train viable fields
            print("3) Train Viable Fields")
            for field in self.fields:
                if self.vectorizers[field]:  # is viable
                    print(f"{field} is a target. Creating classifier...")
                    start_time = time.time()
                    # initialize the sgd classifier
                    self.clfs[field] = self.create_perceptron_classifier()
                    # train the classifier for this field using stored y labels array
                    self.clfs[field].fit(self.data_vectorized[field], self.targets[field])  

                    # self.logger.info('[%s] Trained classifier in %s seconds', field, time.time() - start_time)
                    print(f"[{field}] Trained classifier in {time.time() - start_time} seconds")

            # 4. DONE. We now have classifiers trained for every target field... is the target included??

    def _top_ranked_features(self, field, x_train, y_train):
        '''
        Returns the sorted list of feature names according to statistical score (chi squared test for non-negative features).
        Chi-squared test measures the dependence between stochastic variables. This function `weeds out` 
        features that are likely to be independent of class, therefore irrelevant for clasification
        '''
        ch2 = SelectKBest(chi2, k=25)
        ch2.fit_transform(x_train, y_train)
        top_ranked_features = sorted(enumerate(ch2.scores_), key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)[:25]

        # VECTORIZER SHOULD BE STATEFUL. HASHINGVECTORIZER WON'T WORK.
        feature_names = np.asarray(self.vectorizers[field].named_steps['vectorizer'].get_feature_names())   # only works if vectorizer is stateful

        top_ranked_features_indices = list(map(list, zip(*top_ranked_features)))[0]

        return [{'feature': feature, 'p': (None if math.isnan(pvalue) else pvalue)}
                 for feature, pvalue in zip(
                     feature_names[top_ranked_features_indices],
                     ch2.pvalues_[top_ranked_features_indices])]
    
    def show_top_features(self, field):
        # x_train, x_test, y_train, y_test = model_selection.train_test_split(
        #     self.data_vectorized[field], np.array(self.targets[field]), test_size=0.3, random_state=0)
        
        x_train = self.data_vectorized[field]
        y_train = np.array(self.targets[field])

        return self._top_ranked_features(field, x_train, y_train)
    

    def _get_accuracy(self, y_pred, y_true):
        '''Computes the fraction of correct predictions over total predictions'''
        return np.sum(y_pred == y_true) / len(y_pred)

    def _benchmark(self, clf, x_test, y_test, field):
        '''Performs testing and computes performance metrics given vectorized data and true target values'''
        y_pred = clf.predict(x_test)

        #print(f"Unpredicted labels for [{field}]", set(y_test) - set(y_pred))
        print(f"=========> Accuracy for classfier[{field}]: {100*(self._get_accuracy(y_pred, y_test))}%")
        confident = {}

        best_f_score, best_threshold = 0.0, None

        for threshold in np.arange(0, 4, 0.2):
            n_confident = 0
            n_confident_wrong = 0
            # clf.decision_function returns the confidence score.
            # The confidence score for a sample is proportional to the signed distance of that sample to the hyperplane
            for prediction, actual, confidence in zip(y_pred, y_test, clf.decision_function(x_test)):
                # simulate a decision threshold... it's an added layer for the listo application
                if np.max(confidence) > threshold:
                    # If the most probable class has confidence larger than the minimum threshold, it is a confident prediction
                    n_confident += 1
                    if prediction != actual:
                        n_confident_wrong += 1   # confident predictions that are wrong should be minimized! e.g. false positives

            if n_confident > 0:
                recall = 100.0 * (n_confident - n_confident_wrong) / len(y_test)
                precision = 100.0 * (n_confident - n_confident_wrong) / n_confident
                # not sure about this formula...
                f_score = (1 + BETA ** 2) * (precision * recall) / ((BETA ** 2) * precision + recall)

                if f_score > best_f_score:
                    best_threshold = threshold
                    best_f_score = f_score

                confident[threshold] = {'recall': recall, 'precision': precision, 'f_score': f_score}
        # Precision - measures true positive predictions over positive predictions made (weighted average)
        # Recall - measures true positive predictions over total positive observations (weighted average)
        # f1 - 
        # confident - a dictionary with threshold value -> P,R, and f_scores
        # the best f score and the best threshold corresponding to that score
        return {
            'precision': metrics.precision_score(y_test, y_pred, average='weighted', pos_label=None),
            'recall':    metrics.recall_score(y_test, y_pred, average='weighted', pos_label=None),
            'f1':        metrics.f1_score(y_test, y_pred, average='weighted', pos_label=None),
            'confident': confident,
            'confident_best_f_score': best_f_score,
            'confident_best_threshold': best_threshold
        }

    def benchmark(self, field):
        '''
        Perform training and testing on a given field classifier,
        using a 70-30 split, with a temproary classifier and cached data_vectorized
        '''
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            self.data_vectorized[field], np.array(self.targets[field]), test_size=0.3, random_state=0)

        # Create a temporary classifier for benchmarking
        clf = self.create_log_classifier()
        try:
            #clf.partial_fit(x_train, y_train, classes=set(y_train))
            clf.fit(x_train, y_train)
        except Exception as e:
            print("X____X   error", e)
        return self._benchmark(clf, x_test, y_test, field)
    


'''
OBSERVATIONS WRT CURRENT IMPLEMENTATION

- Data is extracted as input features to predict any other target (all vs all) under a correlation premise (¿).
- there is no pre-selection of inputs and targets. the data we have is the data we use.
- thus, there is a risk that for a given target there is not enough input data, since we can't use null values
- should there be a predefinition for targets? some mapping of f(input fields) -> (target field) instead of f(inputs except target) -> target

'''