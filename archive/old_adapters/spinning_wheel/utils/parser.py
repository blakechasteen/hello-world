# -*- coding: utf-8 -*-
"""
Parser Utilities
================
Helper functions for parsing various input formats.
"""

import json
import csv
from typing import Dict, List, Any
from pathlib import Path


def parse_json(data: Any) -> Dict:
    """
    Parse JSON data from various sources.
    
    Args:
        data: Can be dict, str (JSON string), or Path (file path)
        
    Returns:
        Parsed dict
    """
    if isinstance(data, dict):
        return data
    
    elif isinstance(data, str):
        # Try parsing as JSON string first
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            # Maybe it's a file path
            try:
                with open(data, 'r') as f:
                    return json.load(f)
            except:
                return {}
    
    elif isinstance(data, Path):
        with open(data, 'r') as f:
            return json.load(f)
    
    return {}


def parse_csv(data: Any) -> List[Dict]:
    """
    Parse CSV data from various sources.
    
    Args:
        data: Can be list of dicts, str (CSV string), or Path (file path)
        
    Returns:
        List of dicts (one per row)
    """
    if isinstance(data, list):
        return data
    
    elif isinstance(data, (str, Path)):
        rows = []
        with open(data, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    
    return []