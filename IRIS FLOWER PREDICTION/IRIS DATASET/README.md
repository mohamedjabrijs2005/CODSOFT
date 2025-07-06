# Iris Flower Classification

This project demonstrates how to use the classic Iris dataset to train a machine learning model that classifies iris flowers into three species: setosa, versicolor, and virginica. The script includes data exploration, visualization, model training, evaluation, and a confusion matrix.

## Features
- Data overview and summary
- Visualizations: count plot, pairplot, boxplots, correlation heatmap
- Random Forest classifier for species prediction
- Model evaluation: accuracy, classification report, confusion matrix

## How to Run
1. Ensure you have Python 3.x installed.
2. Install required packages:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
3. Place `IRIS.csv` and `iris_flower_classification.py` in the same directory.
4. Run the script:
   ```bash
   python iris_flower_classification.py
   ```

## Files
- `IRIS.csv`: The dataset file
- `iris_flower_classification.py`: Main Python script
- `iris_flower_classification.ipynb`: Jupyter notebook version (see this repo)

## Dataset
The Iris dataset contains 150 samples with the following columns:
- sepal_length
- sepal_width
- petal_length
- petal_width
- species

## Output
- Visualizations of the dataset
- Model accuracy and classification report
- Confusion matrix heatmap

## License
This project is for educational purposes.
