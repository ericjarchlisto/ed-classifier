import json
import logging
import math
import re
#import resource # Linux specific
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
# from elasticsearch_dsl.search import Search

from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# params
CONFIDENCE_THRESHOLD = 0.6
EXTRACTION_CHUNK_SIZE = 5000
PROCESSING_CHUNK_SIZE = 1000
BETA = 0.25  # Prioritize precision over recall
MAX_CLASSES = 500

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
                all_text = self._convertAccented(all_text.lower(), accent_rgx)

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

    def __init__(self, clf_factory=None, part='all'):
            # self.logger = logging.getLogger(f'{__package__}.{__name__}.simple-classif')
            self.df = pd.read_csv(DATASET_PATH, index_col=0)
            self.prepareDataFrame()

            self.fields = ['nature', 'cost_center']  # targets
            self.targets = {field: [] for field in self.fields}  # field name -> labels values array
            self.vectorizers = {field: None for field in self.fields}  # field name -> sklearn vectorizer v(invoices_with_field) 
            self.data_vectorized = {field: None for field in self.fields}  # field name -> output of vectorizers[field]
            self.last_modified_on = 0
            self.part = part  
            self.is_viable = False   # huh?
            self.clfs = {}  # field name -> sklearn classifier: f(invoices, target_labels_array)
            self.update() # the very first train

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


    def create_sgd_classifier(self):
        '''Create a Stochastic Gradient Descent SVM'''
        # no customization YET
        return SGDClassifier(alpha=1e-5, loss='log', n_jobs=-1) # by default is an SVM
    

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
                vectorized = self.vectorizers[field].transform(invoices_with_field)  # triggers both clean and vectorize steps on IR attribute

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

        chunk: Dict[int, InvoiceRecord] = {}
        for index, hit in tqdm(data.iterrows(), total=len(data)):
            # iterate over the rows of data, named hit
            record = self._process_hit(hit)  # returns InvoiceRecord
            if record is not None:
                chunk[hit['id']] = record # can it be index?

            if len(chunk) >= EXTRACTION_CHUNK_SIZE:

                _process_chunk(chunk)  # vectorize invoices so far and then clear the chunk
        # Leftover results (modulus PROCESSING_CHUNK_SIZE)
        if chunk:
            _process_chunk(chunk)


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
                    self.clfs[field] = self.create_sgd_classifier()
                    # train the classifier for this field using stored y labels array
                    self.clfs[field].fit(self.data_vectorized[field], self.targets[field])  

                    # self.logger.info('[%s] Trained classifier in %s seconds', field, time.time() - start_time)
                    print(f"[{field}] Trained classifier in {time.time() - start_time} seconds")

            # 4. DONE. We now have classifiers trained for every target field... is the target included??

    def _top_ranked_features(self, field, x_train, y_train):
        '''Not sure what this one does yet'''
        ch2 = SelectKBest(chi2, k=25)
        ch2.fit_transform(x_train, y_train)
        top_ranked_features = sorted(enumerate(ch2.scores_), key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)[:25]

        feature_names = np.asarray(self.vectorizers[field].get_feature_names())

        top_ranked_features_indices = list(map(list, zip(*top_ranked_features)))[0]

        return [{'feature': feature, 'p': (None if math.isnan(pvalue) else pvalue)}
                 for feature, pvalue in zip(
                     feature_names[top_ranked_features_indices],
                     ch2.pvalues_[top_ranked_features_indices])]
'''
OBSERVATIONS WRT CURRENT IMPLEMENTATION

- Data is extracted as input features to predict any other target (all vs all) under a correlation premise (¿).
- there is no pre-selection of inputs and targets. the data we have is the data we use.
- thus, there is a risk that for a given target there is not enough input data, since we can't use null values
- should there be a predefinition for targets? some mapping of f(input fields) -> (target field) instead of f(inputs except target) -> target
'''