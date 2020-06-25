#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 00:41:46 2019

@author: amc
"""

import csv
import json
import pandas as pd

with open('script.json') as json_file:  
    data = json.load(json_file)
    
    
parsed_data = pd.DataFrame.from_records(data['rows'])

parsed_data = pd.DataFrame.from_records(parsed_data.doc.values)

parsed_data.to_csv('script_rows.csv')