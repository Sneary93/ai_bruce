config_ini_location = 'config_new.ini'

import configparser

config = configparser.ConfigParser()
config.read(config_ini_location)
openai_api_key = config['OpenAI']['API_KEY']
