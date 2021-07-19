# Car Damage Classification

A small application based on TensorFlow and Keras for classifying car damage properties on images.

## Technologies

- TensorFlow/Keras
- SciKit Learn
- CUDA/cuDNN

## Local Development

### Dependencies
- CUDA compatible graphics card
- CUDA and cuDNN installed
- Access to TF-Hub to fetch pretrained models OR pretrained models locally at ./keras/models/~
- Python Environment configured
  ```sh
  pip install -r requirements.txt
  ```
## Run Car Damage Application
- Adjust application_config.json to point to correct model files
- Update label files accordingly
```sh
  python main.py --config <path/to/your/application_config.json> --debug False
```

# Training Results

Training was done using cross validation. We report accuracy results for each iteration here. 
A more detailed results for each label can be found with each model inside a results.txt file.

## ResNet50

### Damage Severity

Accuracies for cross validation splits: [0.7260273694992065, 0.7424657344818115, 0.7369862794876099, 0.7041096091270447, 0.7178082466125488]
on average: 0.7254794478416443
best iteration: 2

| Label    | CrossVal k = 1  | CrossVal k = 2  | CrossVal k = 3  | CrossVal k = 4  | CrossVal k = 5  |
| -------- | --------------- | --------------- | --------------- | --------------- | --------------- |
|          | Total | Correct | Total | Correct | Total | Correct | Total | Correct | Total | Correct |
| -------- | :---: | :-----: | :---: | :-----: | :---: | :-----: | :---: | :-----: | :---: | :-----: |
| Minor    | 53    | 30      | 53    | 32      | 53    | 32      | 54    | 21      | 54    | 21      |
| Moderate | 58    | 22      | 58    | 15      | 58    | 23      | 57    | 22      | 57    | 22      |
| Severe   | 70    | 42      | 70    | 54      | 70    | 48      | 70    | 47      | 70    | 47      |
| Whole    | 184   | 171     | 184   | 170     | 184   | 166     | 184   | 167     | 184   | 167     |

### Location

Accuracies for cross validation splits: [0.8404109477996826, 0.8472602963447571, 0.8390411138534546, 0.8500000238418579, 0.8547945022583008]
on average: 0.8463013768196106
best iteration: 5

### Damage Type

Accuracies for cross validation splits: [0.8668494820594788, 0.8728767037391663, 0.8586300611495972, 0.8767123222351074, 0.8717808723449707]
on average: 0.8693698883056641
best iteration: 4