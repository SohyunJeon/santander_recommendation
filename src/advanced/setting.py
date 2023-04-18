import configparser
from time import strftime
import os
from pathlib import Path

def generate_config():
    config = configparser.ConfigParser()

    config['system'] = {
        'title': 'Santander Recommendation',
        'version': '0.0.1',
        'update': strftime('%Y-%m-%d %H:%M:%S')
    }

    config['path'] = {
        'data':  './data/',
        'model':  './model/',
    }

    with open('./config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)


def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    system = config['system']
    print(f"{system['title']}: versiong {system['version']} / update {system['update']}")
    return config

if __name__ == '__main__':

    generate_config()
    read_config()



