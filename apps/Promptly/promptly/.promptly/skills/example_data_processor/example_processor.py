#!/usr/bin/env python3
"""
Example data processor script
"""

def process_data(data):
    """Process the input data"""
    results = []
    for item in data:
        processed = {
            'original': item,
            'processed': item.upper(),
            'length': len(item)
        }
        results.append(processed)
    return results

if __name__ == '__main__':
    sample = ['hello', 'world', 'claude']
    print(process_data(sample))
