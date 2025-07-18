# Dataset
## Dataset Format
We use the `LeRobotDataset` format as the default format for storing. LeRobotDataset is a popular format for organizing robot operation trajectories including RGB-D images, instructions, actions, and metadata. The dataset is organized in a directory structure as follows:

```
dataset_root/
├── episodes/
│   ├── episode_000/
│   │   ├── rgb/
│   │   ├── depth/
│   │   ├── actions.jsonl
│   │   └── metadata.json
│   ├── ...
└── dataset_meatadata.json
```

## Convert to LeRobotDataset

## Data Key Remapping

To align the data keys in the dataset with the expected keys in the code, you can customize the data configs under `grmanipulation/configs/data_configs/` ...


## Register a Custom Dataset
To register a custom dataset, you need to create a new Python file in the `grmanipulation/datasets/` directory. The file should define a class that inherits from `LeRobotDataset` and implements the necessary methods for loading and processing the dataset. Here is an example of how to register a custom dataset:

```python
from grmanipulation.datasets import LeRobotDataset

class CustomDataset(LeRobotDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self):
        # Implement your data loading logic here
        pass
```

## Collect Demonstration Dataset in GRUtopia
