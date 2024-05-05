import os
import yaml

class Opt:
    def __init__(self, config_path='config/config.yaml'):
        # Load configuration from YAML file specific to the model name provided
        config = self.read_config(config_path)
        activated_model = config['activated']
        model_config = config[activated_model]

        # Assign model-specific configuration values to class attributes
        self.name_net = activated_model  
        self.learning_rate = model_config['learning_rate']
        self.mean = model_config['mean']
        self.std = model_config['std']
        self.print_freq = model_config['print_freq']
        self.batch_size = model_config['batch_size']
        self.num_workers = model_config['num_workers']
        self.epochs = model_config['epochs']
        self.num_classes = model_config['num_classes']
        self.resize_height = model_config['resize_height']
        self.resize_width = model_config['resize_width']
        self.b_factor = model_config['b_factor']
        self.alpha = model_config['alpha']
        self.load_saved_model = model_config['load_saved_model']
        self.threshold_val_dice = model_config['threshold_val_dice']
        self.path_to_pretrained_model = model_config['path_to_pretrained_model']
        self.project_folder = model_config['project_folder']
        self.data_folder = model_config['data_folder']
        self.results_folder = os.path.join(self.project_folder, model_config['results_folder'])
        
        # Create the results directory if it does not exist
        os.makedirs(self.results_folder, exist_ok=True)

    def read_config(self, path):
        # Read and return the configuration from a YAML file
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config

# Initialize Opt
config_path = 'config/config.yaml' 
opt = Opt(config_path)

print(f"Loaded configuration for model: {opt.name_net}")
