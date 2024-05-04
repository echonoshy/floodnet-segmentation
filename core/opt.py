import os
import yaml

class Opt:
    def __init__(self, config_path='config/config.yaml', model_name='unet'):
        config = self.read_config(config_path)
        
        # Access model-specific configuration
        model_config = config[model_name]

        self.learning_rate = model_config['learning_rate']
        self.mean = model_config['mean']
        self.std = model_config['std']
        self.print_freq = model_config['print_freq']
        self.name_net = model_name  # Use the passed model name directly
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
        
        # Ensure the results folder exists
        os.makedirs(self.results_folder, exist_ok=True)

    def read_config(self, path):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config



# todo: 临时写死
model_name = "unet"
config_path = 'config/config.yaml' 
opt = Opt(config_path, model_name)


print(f"Loaded configuration for model: {opt.name_net}")
