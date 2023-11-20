from social_network_gnn import data
from social_network_gnn import CONFIG

from pathlib import Path

if __name__ == "__main__":
    temp = data.Graph(Path(CONFIG.locations.dataset_path))