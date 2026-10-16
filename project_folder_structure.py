from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(r"D:\AI Projects store\fraud-detection-mlops")


# ============================================================
# FOLDERS TO CREATE
# ============================================================

DIRECTORIES = [

    # ---------------- FRONTEND ----------------
    "frontend",
    "frontend/public",
    "frontend/src",
    "frontend/src/api",
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/charts",
    "frontend/src/hooks",
    "frontend/src/utils",

    # ---------------- BACKEND ----------------
    "backend",
    "backend/app",
    "backend/app/config",
    "backend/app/api",
    "backend/app/api/routes",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/database",
    "backend/app/ml",
    "backend/app/utils",
    "backend/tests",

    # ---------------- ML ----------------
    "ml",
    "ml/data",
    "ml/data/ingestion",
    "ml/data/validation",
    "ml/data/preprocessing",
    "ml/features",
    "ml/models",
    "ml/experiments",

    # ---------------- DATA ----------------
    "data",
    "data/raw",
    "data/processed",
    "data/reference",

    # ---------------- MODELS ----------------
    "models",
    "models/production",
    "models/candidates",

    # ---------------- MLOPS ----------------
    "mlops",
    "mlops/monitoring",
    "mlops/monitoring/reports",
    "mlops/retraining",

    # ---------------- NOTEBOOKS ----------------
    "notebooks",

    # ---------------- TESTS ----------------
    "tests",
    "tests/unit",
    "tests/integration",

    # ---------------- SCRIPTS ----------------
    "scripts",

    # ---------------- ARTIFACTS ----------------
    "artifacts",
    "artifacts/metrics",
    "artifacts/plots",
    "artifacts/reports",

    # ---------------- CONFIGS ----------------
    "configs",

    # ---------------- GITHUB ----------------
    ".github",
    ".github/workflows",
]


# ============================================================
# FILES TO CREATE
# ============================================================

FILES = [

    # ---------------- ROOT FILES ----------------
    "README.md",
    ".gitignore",
    ".env.example",
    "docker-compose.yml",
    "Dockerfile",
    "params.yaml",
    "dvc.yaml",

    # ---------------- FRONTEND ----------------
    "frontend/vite.config.js",
    "frontend/index.html",

    "frontend/src/main.jsx",
    "frontend/src/App.jsx",
    "frontend/src/index.css",

    # Frontend API
    "frontend/src/api/axios.js",
    "frontend/src/api/predictionApi.js",
    "frontend/src/api/transactionApi.js",
    "frontend/src/api/analyticsApi.js",
    "frontend/src/api/monitoringApi.js",
    "frontend/src/api/modelApi.js",

    # Frontend components
    "frontend/src/components/Navbar.jsx",
    "frontend/src/components/Sidebar.jsx",
    "frontend/src/components/StatCard.jsx",
    "frontend/src/components/PredictionCard.jsx",
    "frontend/src/components/RiskBadge.jsx",
    "frontend/src/components/Loading.jsx",
    "frontend/src/components/ErrorMessage.jsx",

    # Frontend pages
    "frontend/src/pages/Dashboard.jsx",
    "frontend/src/pages/PredictFraud.jsx",
    "frontend/src/pages/Transactions.jsx",
    "frontend/src/pages/Analytics.jsx",
    "frontend/src/pages/Monitoring.jsx",
    "frontend/src/pages/ModelManagement.jsx",

    # Frontend charts
    "frontend/src/charts/FraudTrendChart.jsx",
    "frontend/src/charts/TransactionChart.jsx",
    "frontend/src/charts/ModelMetricsChart.jsx",
    "frontend/src/charts/DriftChart.jsx",

    # Frontend hooks
    "frontend/src/hooks/usePrediction.js",
    "frontend/src/hooks/useTransactions.js",
    "frontend/src/hooks/useMonitoring.js",

    # Frontend utilities
    "frontend/src/utils/formatters.js",
    "frontend/src/utils/constants.js",

    # ---------------- BACKEND ----------------
    "backend/requirements.txt",
    "backend/Dockerfile",

    "backend/app/__init__.py",
    "backend/app/main.py",

    # Backend config
    "backend/app/config/__init__.py",
    "backend/app/config/settings.py",

    # Backend API
    "backend/app/api/__init__.py",
    "backend/app/api/dependencies.py",

    "backend/app/api/routes/__init__.py",
    "backend/app/api/routes/prediction.py",
    "backend/app/api/routes/transactions.py",
    "backend/app/api/routes/analytics.py",
    "backend/app/api/routes/monitoring.py",
    "backend/app/api/routes/model.py",

    # Backend schemas
    "backend/app/schemas/__init__.py",
    "backend/app/schemas/prediction.py",
    "backend/app/schemas/transaction.py",
    "backend/app/schemas/analytics.py",
    "backend/app/schemas/monitoring.py",
    "backend/app/schemas/model.py",

    # Backend services
    "backend/app/services/__init__.py",
    "backend/app/services/prediction_service.py",
    "backend/app/services/transaction_service.py",
    "backend/app/services/analytics_service.py",
    "backend/app/services/monitoring_service.py",
    "backend/app/services/model_service.py",

    # Backend database
    "backend/app/database/__init__.py",
    "backend/app/database/database.py",
    "backend/app/database/models.py",
    "backend/app/database/crud.py",

    # Backend ML
    "backend/app/ml/__init__.py",
    "backend/app/ml/model_loader.py",
    "backend/app/ml/predictor.py",
    "backend/app/ml/preprocessor.py",

    # Backend utilities
    "backend/app/utils/__init__.py",
    "backend/app/utils/logger.py",
    "backend/app/utils/exceptions.py",

    # Backend tests
    "backend/tests/test_api.py",

    # ---------------- ML ----------------

    # Data
    "ml/data/ingestion.py",
    "ml/data/validation.py",
    "ml/data/preprocessing.py",

    # Features
    "ml/features/feature_engineering.py",

    # Models
    "ml/models/train.py",
    "ml/models/evaluate.py",
    "ml/models/tune.py",

    # Experiments
    "ml/experiments/baseline.py",
    "ml/experiments/random_forest.py",
    "ml/experiments/xgboost.py",

    # ---------------- MLOPS ----------------

    # Monitoring
    "mlops/monitoring/data_drift.py",
    "mlops/monitoring/prediction_drift.py",
    "mlops/monitoring/performance.py",

    # Retraining
    "mlops/retraining/trigger.py",
    "mlops/retraining/retrain.py",
    "mlops/retraining/validation.py",

    # ---------------- NOTEBOOKS ----------------
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocessing.ipynb",
    "notebooks/03_feature_engineering.ipynb",
    "notebooks/04_model_training.ipynb",
    "notebooks/05_evaluation.ipynb",

    # ---------------- TESTS ----------------
    "tests/unit/test_ml_pipeline.py",
    "tests/integration/test_integration.py",

    # ---------------- SCRIPTS ----------------
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/retrain.py",

    # ---------------- CONFIGS ----------------
    "configs/model.yaml",
    "configs/features.yaml",
    "configs/monitoring.yaml",

    # ---------------- GITHUB ----------------
    ".github/workflows/ci.yml",
    ".github/workflows/training.yml",
]


# ============================================================
# CREATE FOLDERS
# ============================================================

print("Creating project folders...")

for directory in DIRECTORIES:
    folder_path = PROJECT_ROOT / directory
    folder_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# CREATE FILES
# ============================================================

print("Creating project files...")

for file in FILES:
    file_path = PROJECT_ROOT / file
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        file_path.touch()


# ============================================================
# CREATE .gitkeep FILES
# ============================================================

GITKEEP_DIRECTORIES = [
    "data/raw",
    "data/processed",
    "data/reference",
    "models/production",
    "models/candidates",
    "mlops/monitoring/reports",
    "artifacts/metrics",
    "artifacts/plots",
    "artifacts/reports",
]

for directory in GITKEEP_DIRECTORIES:
    gitkeep_path = PROJECT_ROOT / directory / ".gitkeep"
    gitkeep_path.touch(exist_ok=True)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("PROJECT STRUCTURE CREATED SUCCESSFULLY!")
print("=" * 60)
print(f"Project location: {PROJECT_ROOT}")
print()
print("You can now open the project folder in VS Code.")