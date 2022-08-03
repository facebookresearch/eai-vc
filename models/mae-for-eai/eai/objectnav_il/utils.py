import json


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))
