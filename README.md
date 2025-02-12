# Reinhard It 

*This repository contains code for comparing the impact of Reinhard normalization on a dataset with HER2-stained images classified into four categories: 0, 1+, 2+, and 3+. The code supports training in two modes: with and without normalization.*

## 🚀 Running the Code

Start by installing the dependencies:

```bash
pip install -r requirements.txt
```

Then, simply run:
```python
python3 testing_normalization.py
```

### 🔧 Additional Configurations
Arguments (Optional – default values are already assigned):
* ```--mode```: Specifies the training mode [Reinhard or without the technique].
* ```--num_epochs```: Defines the number of epochs.

Dataset
* Make sure to set the dataset directory path in the data_path variable (For dataset requests, please contact us).


## 📁 File Structure
```
matrix_images/ -> Images used for normalization matrix
./
 ├── testing_normalization.py -> Main script for execution
 ├── generate_matrix.py -> Generates the normalization matrix
 ├── preprocessing.py -> Splits the dataset into training and testing
 ├── reinhard.py -> Normalization algorithm
 ├── test.py -> Computes performance metrics after execution
 ├── train.py -> Model training script
 ├── transformed_dataset.py -> Handles dataset transformations
```

## 📊 Results Overview

| Model                          | F1 Score      | Recall        | Precision     |
|--------------------------------|--------------|--------------|--------------|
| Without Normalization          | 0.89 ± 0.0   | 0.90 ± 0.0   | 0.89 ± 0.0   |
| With Reinhard Normalization    | 0.91 ± 0.0   | 0.92 ± 0.0   | 0.91 ± 0.0   |

## 📄 Paper

```
@article{ReinhardIt,
  todo
}
```

