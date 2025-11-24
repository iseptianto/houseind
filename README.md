# 🏠 House Price Predictor – An MLOps Learning Project

Welcome to the **House Price Predictor** project! This is a real-world, end-to-end MLOps use case designed to help users, marketing teams, and stakeholders make informed decisions in predicting house prices in Indonesia.

This project aims to assist in decision-making processes for property pricing by providing accurate price predictions based on various property features. The dataset uses OLX.co.id as a reference source, and there's significant potential for improvement by incorporating additional housing price data from other sources to provide more diverse and interesting model variations.

**Key Notes:**
- Currently covers 21 provinces in Indonesia that actively list houses/apartments on OLX
- Needs additional data from the remaining 17 provinces (total 38 provinces in Indonesia) to create a more comprehensive and attractive model

You'll start from raw data and move through data preprocessing, feature engineering, experimentation, model tracking with MLflow, and optionally using Jupyter for exploration – all while applying industry-grade tooling.


## 📦 Project Structure

```
house-price-predictor/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD pipelines
├── app/
│   └── fastapi_app/        # Legacy FastAPI application structure
├── configs/                # YAML-based configuration for models
├── data/                   # Raw and processed datasets
│   ├── mlflow/            # MLflow tracking database
│   ├── processed/         # Cleaned and engineered datasets
│   └── raw/               # Original raw data files
├── deployment/
│   ├── kubernetes/        # Kubernetes deployment manifests
│   └── mlflow/            # Docker Compose setup for MLflow
├── fastapi_app/           # Legacy FastAPI app structure
├── models/                # Trained models and preprocessors
│   ├── trained/           # Legacy model storage
│   ├── modelbaru.pkl      # New trained model
│   ├── barupreprocessor.pkl # New preprocessor
│   └── model_config.yaml  # Model configuration
├── notebooks/             # Jupyter notebooks for experimentation
├── src/
│   ├── api/               # FastAPI application code
│   │   ├── inference.py   # Model inference logic
│   │   ├── main.py        # FastAPI app entry point
│   │   ├── schemas.py     # Pydantic models
│   │   └── utils.py       # Utility functions
│   ├── data/              # Data cleaning and preprocessing scripts
│   ├── features/          # Feature engineering pipeline
│   └── models/            # Model training and evaluation
├── streamlit_app/         # Streamlit web application
│   ├── app.py             # Main Streamlit app
│   ├── Dockerfile         # Streamlit container config
│   └── translations.py    # Multi-language support
├── tests/                 # Unit and integration tests
├── training/              # Training pipeline scripts
├── create_new_model.py    # Script to create new models
├── docker-compose.yaml    # Multi-service orchestration
├── Dockerfile             # FastAPI container config
├── final.csv              # Dataset for Streamlit app
├── requirements.txt       # Python dependencies
└── README.md              # You’re here!
```

---

## 🛠️ Setting up Learning/Development Environment

To begin, ensure the following tools are installed on your system:

- [Python 3.11](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- [Visual Studio Code](https://code.visualstudio.com/) or your preferred editor
- [UV – Python package and environment manager](https://github.com/astral-sh/uv)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) **or** [Podman Desktop](https://podman-desktop.io/)

---

## 🚀 Preparing Your Environment

1. **Fork this repo** on GitHub.

2. **Clone your forked copy:**

   ```bash
   # Replace xxxxxx with your GitHub username or org
   git clone https://github.com/xxxxxx/house-price-predictor.git
   cd house-price-predictor
   ```

3. **Setup Python Virtual Environment using UV:**

   ```bash
   uv venv --python python3.11
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

---

## 📊 Setup MLflow for Experiment Tracking

To track experiments and model runs:

```bash
cd deployment/mlflow
docker compose -f mlflow-docker-compose.yml up -d
docker compose ps
```

> 🐧 **Using Podman?** Use this instead:

```bash
podman compose -f mlflow-docker-compose.yml up -d
podman compose ps
```

Access the MLflow UI at [http://localhost:5555](http://localhost:5555)

---

## 📒 Using JupyterLab (Optional)

If you prefer an interactive experience, launch JupyterLab with:

```bash
uv python -m jupyterlab
# or
python -m jupyterlab
```

---

## 🔁 Model Workflow

### 🧹 Step 1: Data Processing

Clean and preprocess the raw housing dataset:

```bash
python src/data/run_processing.py   --input data/raw/house_data.csv   --output data/processed/cleaned_house_data.csv
```

---

### 🧠 Step 2: Feature Engineering

Apply transformations and generate features:

```bash
python src/features/engineer.py   --input data/processed/cleaned_house_data.csv   --output data/processed/featured_house_data.csv   --preprocessor models/trained/preprocessor.pkl
```

---

### 📈 Step 3: Modeling & Experimentation

Train your model and log everything to MLflow:

```bash
python src/models/train_model.py   --config configs/model_config.yaml   --data data/processed/featured_house_data.csv   --models-dir models   --mlflow-tracking-uri http://localhost:5555
```

---


## Building FastAPI and Streamlit

The code for both applications is available in `src/api` and `streamlit_app`. To build and launch these apps:

### Quick Start with Docker Compose

```bash
# Build and run all services
docker compose up --build

# Or run in background
docker compose up -d --build

# View running services
docker compose ps

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Manual Setup

1. **FastAPI Application** (`src/api/`)
   - Dockerfile is available in the root directory
   - Run locally: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`

2. **Streamlit Application** (`streamlit_app/`)
   - Dockerfile is available in `streamlit_app/`
   - Run locally: `cd streamlit_app && streamlit run app.py`
   - Set `API_URL=http://fastapi:8000` in environment for Streamlit to connect to FastAPI

### Testing the API

Once both apps are running, you can access the Streamlit web UI and make predictions.

You can also test predictions with FastAPI directly using:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "LB": 120,
  "LT": 150,
  "KM": 2,
  "KT": 3,
  "Provinsi": "Jawa Barat",
  "Kota/Kab": "Bandung",
  "Type": "rumah"
}'
```

**Note:** Replace `http://localhost:8000/predict` with the actual endpoint URL based on your deployment.

---

## 🐛 Troubleshooting Guide

### Common Issues and Solutions

#### 1. **Model/Model Loading Errors**
```
FileNotFoundError: Model file not found: /models/trained/house_price_best.pkl
```
**Solution:**
- Ensure you've trained the model first: `python src/models/train_model.py`
- Check that model files exist in `models/trained/`
- Verify Docker volume mounting if using containers

#### 2. **CSV Loading Errors in Streamlit**
```
Kolom CSV kurang: {'Provinsi', 'Kota/Kab'}. Pastikan ada 'Provinsi' dan 'Kota/Kab'.
```
**Solution:**
- Ensure `final.csv` exists in the project root or set `CSV_PATH` environment variable
- Check CSV column names match expected format (case-sensitive)
- Verify CSV file encoding (should be UTF-8)

#### 3. **API Connection Errors**
```
Failed to establish connection to FastAPI server
```
**Solution:**
- Ensure FastAPI is running: `docker compose up fastapi`
- Check API URL in Streamlit settings (default: `http://localhost:8000`)
- Verify network connectivity in Docker Compose

#### 4. **Port Conflicts**
```
Port already in use: 8501 or 8000
```
**Solution:**
- Stop other services using these ports
- Modify ports in `docker-compose.yaml` if needed
- Use `docker compose down` to stop all services

#### 5. **MLflow Connection Issues**
```
Failed to connect to MLflow tracking server
```
**Solution:**
- Start MLflow: `cd deployment/mlflow && docker compose up -d`
- Check MLflow URL (default: `http://localhost:5000`)
- Verify network connectivity

#### 6. **Environment Variable Issues**
```
Invalid API_URL format: must start with http:// or https://
```
**Solution:**
- Set proper environment variables:
  ```bash
  export API_URL="http://localhost:8000"
  export CSV_PATH="final.csv"
  ```

### Development Setup

#### Local Development
```bash
# 1. Install dependencies
uv venv --python python3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Start MLflow
cd deployment/mlflow
docker compose up -d

# 3. Train model
python src/models/train_model.py --config configs/model_config.yaml --data data/processed/final.csv --models-dir models --mlflow-tracking-uri http://localhost:5000

# 4. Start FastAPI
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Start Streamlit (in another terminal)
cd streamlit_app
streamlit run app.py
```

#### Docker Development
```bash
# Build and run all services
docker compose up --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Testing

```bash
# Run API tests
pytest tests/test_api.py -v

# Run inference tests
pytest tests/test_inference.py -v

# Run all tests
pytest tests/ -v
```

### Performance Optimization

- **Model Loading**: Models are cached after first load
- **Feature Engineering**: Optimized pandas operations
- **API**: Async endpoints with proper error handling
- **Streamlit**: Efficient data loading with caching

### 🤝 Contributing

We welcome contributions, issues, and suggestions to make this project even better. Feel free to fork, explore, and raise PRs!

### 📊 Dataset Enhancement Opportunities

To improve model accuracy and coverage:

- **Add more provinces**: Currently covers 21 provinces, needs 17 more for complete Indonesia coverage
- **Additional data sources**: Incorporate data from other property platforms beyond OLX
- **Real-time data**: Implement data pipelines for continuous updates
- **Property features**: Add more features like year built, condition, amenities, etc.

### 🔄 Future Improvements

- Multi-language support expansion
- Advanced model architectures (neural networks, ensemble methods)
- Real-time prediction API with caching
- Mobile application development
- Integration with property listing platforms

---

Happy Learning!

## 📞 Contact

**Indra Septianto**
- Instagram: [@iseptianto](https://instagram.com/iseptianto)
- LinkedIn: [indra-septianto17](https://www.linkedin.com/in/indraseptianto17/)

