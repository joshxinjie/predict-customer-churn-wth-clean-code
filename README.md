# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we identify credit card customers that are most likely to 
churn. The completed project will include a Python package for a machine 
learning project that follows coding (PEP8) and engineering best practices for 
implementing software (modular, documented, and tested). The package will also 
have the flexibility of being run interactively or from the command-line 
interface (CLI).

## Files and data description

```
.
├── Guide.ipynb          # Given: Getting started and troubleshooting tips
├── churn_notebook.ipynb # Given: Contains the code to be refactored
├── churn_library.py     # Functions for the churn model
├── test_churn_library.py # Unit tests for churn_library.py
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── results             # Store model classification results 
│   ├── lr_test_report.csv
│   ├── lr_train_report.csv
│   ├── rf_test_report.csv
│   └── rf_test_report.csv
├── logs				 # Store logs
└── models               # Store models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
```

## Running Files
How do you run your files? What should happen when you run your files?

1. Create a new conda enviroment, e.g.
```
conda create -n pred_cust_churn python=3.8
```

2. Activate the environment, e.g.
```
conda activate pred_cust_churn
```

3. Install the dependencies
```
python -m pip install -r requirements_py3.8.txt
```

4. Run the tests
```
pytest
```

5. View the tests log file at `logs/churn_library.log`