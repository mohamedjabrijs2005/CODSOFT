import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate synthetic movie data
np.random.seed(42)
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
directors = ['Spielberg', 'Nolan', 'Tarantino', 'Scorsese', 'Kubrick']
actors = ['DiCaprio', 'Johansson', 'Pitt', 'Streep', 'Hanks']

data = {
    'genre': np.random.choice(genres, 200),
    'director': np.random.choice(directors, 200),
    'actor': np.random.choice(actors, 200),
    'budget_million': np.random.uniform(10, 200, 200),
    'duration_min': np.random.randint(80, 180, 200),
    'rating': np.random.uniform(4, 9, 200)  # Simulated ratings
}
df = pd.DataFrame(data)

# 2. Feature and target split
X = df.drop('rating', axis=1)
y = df['rating']

# 3. Preprocessing pipeline
categorical_features = ['genre', 'director', 'actor']
numerical_features = ['budget_million', 'duration_min']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# 4. Model pipeline
model = Pipeline([
    ('pre', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the model
model.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 8. Predict a new movie rating
new_movie = pd.DataFrame([{
    'genre': 'Action',
    'director': 'Nolan',
    'actor': 'DiCaprio',
    'budget_million': 150,
    'duration_min': 130
}])
predicted_rating = model.predict(new_movie)
print("Predicted rating for new movie:", predicted_rating[0])