import json
import logging
import math
import re
import resource
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
    counterparty_alias: str
    counterparty_rfc: str
    descriptions: str
    # no 'text', no 'prepayment as inputs anymore

@dataclass
class InvoiceRecord(InvoiceRecordForClassification):
    id: int
    values: Dict[str, Any]  # to know wich fields are populated


class PartSelector(BaseEstimator, TransformerMixin):
    """Extract and transform text from each document for Vectorizing
    
        implements fit() and transform() according to sklearn API conventions
    """
    def __init__(self, part):
        self.part = part

    def fit(self, x, y=None):
        # no fitting required
        return self

    def transform(self, invoices: List[InvoiceRecordForClassification]):
        # For every invoice's text features convert into vectorizable plain text
        # This is the 1st step in the vectorizer pipeline. 2nd is Hashingvectorizer
        
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
    '''

    clfs: Dict[str, SGDClassifier]  # field name -> sklearn SVM Classifier
    vectorizers: Dict[str, Pipeline]  # field name -> Transformer-Vectorizer object
    targets: Dict[str, List[str]]   # field name -> [label values]
    last_modified_on: int  # UTC timestamp, for selecting the data to use for partial fit

    def __init__(self, clf_factory=None, part='all'):
            self.df = pd.read_csv(DATASET_PATH, index_col=0)

            self.fields = ['cost_center', 'nature']
            self.targets = {field: [] for field in self.fields}
            self.vectorizers = {field: None for field in self.fields}
            self.data_vectorized = {field: None for field in self.fields}
            self.last_modified_on = 0
            self.part = part  # i thought part was field-unique
            self.logger = logging.getLogger(f'{__package__}.{__name__}.{customer.id}.{workflow_name}')
            self.is_viable = False   # huh?
            self.clfs = {}
            self.update() # the very first train

    def create_sdg_classifier(self):
        # no customization YET
        return SGDClassifier(alpha=1e-5, loss='log', n_jobs=-1) # by default is an SVM
    

    def get_classes(self, field:str) -> Set[str]:
        if field in self.df:
            return set(self.df[field].unique())

        return set()


    def update(self):
            """Process next batch of invoices (based on categorization time),
            re-train the classifier using the whole set"""

            if not self.fields:
                return

            # 1. Determine viability of a field's classes
            for field in self.fields:
                classes = self.get_classes(field)
                self.logger.info('[%s] Extracted %s unique classes', field, len(classes))
                self.logger.info('[%s] %s', field, repr(classes))

                if 2 <= len(classes) <= MAX_CLASSES:
                    self.is_viable = True
                    part = self.workflow.fields[field]['classifier'].get('part', 'all')
                    self.vectorizers[field] = Pipeline([
                        ('selector', PartSelector(part)),
                        ('vectorizer',
                        HashingVectorizer(n_features=262144, ngram_range=(1, 2),
                                        binary=True, strip_accents='ascii'))
                    ])
                else:
                    self.logger.info('[%s] Not viable: %s classes', field, len(classes))

            # 2. Extract data for viable fields
            start_time = time.time()
            self.extract_data()
            self.logger.info('Extracted data in %s seconds', time.time() - start_time)

            # 3. Train viable fields
            for field in self.fields:
                if self.vectorizers[field]:  # is viable
                    start_time = time.time()
                    self.clfs[field] = self.create_classifier()
                    self.clfs[field].fit(self.data_vectorized[field], self.targets[field])
                    self.logger.info('[%s] Trained classifier in %s seconds', field, time.time() - start_time)