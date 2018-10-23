import json

def yield_json_dicts(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)
