import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import sys

THRESHOLD = 5000.0

# Cargar datos
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar último modelo registrado de MLflow
experiment_name = "CI-CD-Lab2"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"❌ Experimento {experiment_name} no existe.")
    sys.exit(1)

# Tomar el último run
from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
if not runs:
    print(f"❌ No hay runs para el experimento {experiment_name}.")
    sys.exit(1)

run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Predicción y validación
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE: {mse:.4f} (umbral: {THRESHOLD})")

if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo no cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)
