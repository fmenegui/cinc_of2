from media.helpers.load_config_module import load_config_module

def train_dx(config_path):
    config = load_config_module(config_path)
    epochs = config.epochs
    fun = config.preprocess
    print(f"Epochs: {epochs}")
    # Example of using the function from config
    result = fun(10)
    print(f"Result of preprocess function: {result}")
    # Insert your training logic here

# Example usage
if __name__ == "__main__":
    config_path = '/home/fdias/repositorios/media/media/train/dummy_config.py'
    train_dx(config_path)

# def train_dx(config):
#     pass

# def run_dx(config, record):
#     pass