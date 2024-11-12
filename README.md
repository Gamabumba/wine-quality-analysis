# wine-quality-analysis

# Analysis and forecasting of wine quality based on Wine Enthusiast data

This project analyzes the Wine Enthusiast dataset to evaluate wine quality based on various attributes, including country, price, and grape variety. The project also involves setting up several machine learning models to predict wine scores.  

**link to kaggle dataset**  
https://www.kaggle.com/datasets/zynicide/wine-reviews


## Project structure

- `data/`: Directory for storing data  
- `notebooks/`: Basic notebook with full data analysis. 
- `reports/`: Generated reports, for example the HTML report from ydata-profiling.  
- `README.md`: Current description of the project.  
- `requirements.txt`: All project dependencies.  
- `.gitignore`: Ignored files such as data and temporary files.  

## Libraries  

- **Pandas**, **Numpy** for working with data  
- **Scikit-learn** for preprocessing and metrics  
- **Matplotlib**, **Seaborn** for visualizations  
- **Optuna** for tuning model hyperparameters  
- **XGBoost**, **LightGBM**, **CatBoost** for modeling  
- **SHAP** for model interpretation  

## Setting up and launching the project  

Clone the repository and install dependencies:
   ```bash
   git clone [<ссылка на репозиторий>](https://github.com/Gamabumba/wine-quality-analysis).git
   cd wine-quality-analysis
   pip install -r requirements.txt
