import yaml
import os

class Config:
    def __init__(self,config_folder,config_file) -> None:
        path = os.path.join(config_folder,config_file)
        with open(path,'r') as file:
            self._config = yaml.safe_load(file)
    
    def __getattr__(self, property_name):
        if (property_name not in self._config.keys()):
            return None
        return self._config[property_name]

        