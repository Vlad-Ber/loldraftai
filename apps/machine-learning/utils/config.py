import json


class TrainingConfig:
    def __init__(self):
        # Default values
        self.num_epochs = 100
        self.embed_dim = 64
        self.num_heads = 8
        self.num_transformer_layers = 2
        self.dropout = 0.1
        self.weight_decay = 0.01
        self.learning_rate = 1e-3
        self.max_grad_norm = 1.0
        self.accumulation_steps = 1
        self.mask_champions = 0.1

        self.calculate_val_loss = True
        self.calculate_val_win_prediction_only = False
        self.log_wandb = True

    def update_from_json(self, json_file: str):
        with open(json_file, "r") as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter '{key}'")

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

    def to_dict(self) -> dict:
        return {key: value for key, value in vars(self).items() if key != "log_wandb"}
