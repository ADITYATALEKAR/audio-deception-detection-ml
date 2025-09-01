# Audio Deception Detection System - Full Architecture Breakdown

## System Overview

The Audio Deception Detection System is a machine learning pipeline designed to classify speech recordings as truthful or deceptive based on acoustic features. The architecture follows a traditional ML pipeline with preprocessing, feature engineering, model training, and evaluation components.

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Raw Audio      │───▶│  Data            │───▶│  Feature        │───▶│  Model Training  │
│  Input          │    │  Preprocessing   │    │  Extraction     │    │  & Evaluation    │
│  (WAV files)    │    │                  │    │                 │    │                  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                               │                         │                         │
                               ▼                         ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
                       │  Data            │    │  Feature        │    │  Model           │
                       │  Augmentation    │    │  Engineering    │    │  Optimization    │
                       │                  │    │                 │    │                  │
                       └──────────────────┘    └─────────────────┘    └──────────────────┘
```

## Detailed Component Architecture

### 1. Data Input Layer

#### 1.1 Raw Data Structure
```
Input:
├── Audio Files (.wav format)
│   ├── Variable length recordings
│   ├── Different sample rates
│   └── Multiple languages (Hindi, English, Bengali)
└── Metadata (CSV)
    ├── filename: Audio file identifier
    ├── Language: Recording language
    └── Story_type: Classification label (true_story/deceptive_story)
```

#### 1.2 Data Validation
- **File Format Check**: Ensures all inputs are valid WAV files
- **Metadata Integrity**: Validates filename matching and label consistency
- **Audio Quality**: Basic checks for corrupted or empty files

### 2. Data Preprocessing Layer

#### 2.1 Audio Standardization Module
```python
class AudioPreprocessor:
    def __init__(self, target_duration=30, sample_rate=22050):
        self.target_duration = target_duration
        self.sample_rate = sample_rate
    
    def trim_audio(self, audio_path, output_path):
        # Standardize to 30-second duration
        # Convert to consistent format (WAV)
        # Normalize audio levels
```

**Processing Steps:**
1. **Duration Normalization**: Trim all recordings to exactly 30 seconds
2. **Format Conversion**: Convert to WAV format if necessary
3. **Sample Rate Standardization**: Resample to consistent rate (22,050 Hz)
4. **Amplitude Normalization**: Normalize audio levels to prevent bias

#### 2.2 Data Augmentation Module
```python
class AudioAugmentor:
    def __init__(self):
        self.augmentation_types = ['noise', 'pitch_shift', 'time_stretch']
    
    def augment_dataset(self, audio_files):
        # Apply multiple augmentation techniques
        # Generate synthetic variations
        # Maintain label consistency
```

**Augmentation Techniques:**
- **Noise Addition**: Add background noise at -30dB
- **Pitch Shifting**: Adjust pitch by ±2 semitones
- **Time Stretching**: Modify speed by ±20%

### 3. Feature Extraction Layer

#### 3.1 Acoustic Feature Extractor
```python
class FeatureExtractor:
    def __init__(self):
        self.features = ['mfcc', 'melspectrogram', 'spectral_rolloff']
        self.feature_dim = 142  # Total feature vector size
    
    def extract_features(self, audio_path):
        # Load audio with librosa
        # Extract multiple feature types
        # Combine into single feature vector
```

#### 3.2 Feature Types and Dimensions

**MFCC (Mel Frequency Cepstral Coefficients)**
- **Dimensions**: 13 coefficients
- **Purpose**: Captures spectral envelope characteristics
- **Computation**: `librosa.feature.mfcc(n_mfcc=13)`
- **Output**: Mean values across time frames

**Mel-Spectrogram**
- **Dimensions**: 128 frequency bins (default)
- **Purpose**: Time-frequency representation in perceptually relevant scale
- **Computation**: `librosa.feature.melspectrogram()`
- **Output**: Mean energy values across time frames

**Spectral Roll-off**
- **Dimensions**: 1 coefficient
- **Purpose**: Frequency distribution measurement
- **Computation**: `librosa.feature.spectral_rolloff()`
- **Output**: Mean roll-off frequency

**Combined Feature Vector**: 13 + 128 + 1 = 142 dimensions

#### 3.3 Feature Engineering Pipeline
```
Raw Audio (30s) → Librosa Load → Feature Extraction → Feature Combination → Normalization
     │                 │              │                    │                │
     │                 │              │                    │                └── StandardScaler (optional)
     │                 │              │                    └── np.hstack([mfcc, mel, rolloff])
     │                 │              └── Extract MFCC, Mel-spec, Roll-off
     │                 └── librosa.load(duration=30, sr=22050)
     └── Input: WAV file path
```

### 4. Machine Learning Layer

#### 4.1 Model Architecture
```python
class DeceptionDetectionModel:
    def __init__(self):
        self.base_model = RandomForestClassifier()
        self.label_encoder = LabelEncoder()
        self.hyperparams = {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_split': 2,
            'random_state': 42
        }
```

#### 4.2 Random Forest Architecture Details

**Ensemble Structure:**
- **Number of Trees**: 300 decision trees
- **Max Depth**: Limited to 10 levels to prevent overfitting
- **Split Criteria**: Gini impurity for classification
- **Bootstrap Sampling**: Each tree trained on random subset of data
- **Feature Sampling**: √(total_features) features considered per split

**Decision Tree Structure:**
```
Root Node (142 features available)
├── Split on best feature (e.g., MFCC_1 < threshold)
│   ├── Left Child (continue splitting)
│   └── Right Child (continue splitting)
└── Leaf Nodes (class predictions: 0=deceptive, 1=true)
```

#### 4.3 Training Pipeline
```
Training Data (80%) → Label Encoding → Model Training → Hyperparameter Tuning
     │                      │               │                    │
     │                      │               │                    └── GridSearchCV
     │                      │               └── RandomForest.fit()
     │                      └── LabelEncoder.fit_transform()
     └── Stratified Split

Validation Data (20%) → Model Evaluation → Performance Metrics
     │                       │                    │
     │                       │                    └── Accuracy, Precision, Recall, F1
     │                       └── Model.predict()
     └── Hold-out for testing
```

### 5. Model Optimization Layer

#### 5.1 Hyperparameter Tuning Architecture
```python
class ModelOptimizer:
    def __init__(self):
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        self.cv_folds = 3
        self.scoring = 'accuracy'
    
    def optimize_model(self, X_train, y_train):
        # GridSearchCV with cross-validation
        # Return best parameters and model
```

#### 5.2 Cross-Validation Strategy
```
Training Set → 3-Fold Cross-Validation → Best Parameters
     │               │                         │
     │               ├── Fold 1 (train/val)    │
     │               ├── Fold 2 (train/val)    │
     │               └── Fold 3 (train/val)    │
     │                                         │
     └── 27 parameter combinations tested ────┘
```

### 6. Evaluation Layer

#### 6.1 Performance Metrics Architecture
```python
class ModelEvaluator:
    def __init__(self):
        self.metrics = [
            'accuracy_score',
            'classification_report',
            'confusion_matrix'
        ]
    
    def evaluate_model(self, model, X_test, y_test):
        # Generate predictions
        # Calculate metrics
        # Create visualizations
```

#### 6.2 Evaluation Metrics Structure
```
Model Predictions → Metric Calculation → Performance Analysis
       │                    │                     │
       │                    ├── Accuracy          │
       │                    ├── Precision         │
       │                    ├── Recall            │
       │                    ├── F1-Score          │
       │                    └── Confusion Matrix  │
       │                                          │
       └── Binary Classification Output ─────────┘
```

## Data Flow Architecture

### Complete Pipeline Flow
```
┌──────────────┐
│ Raw Audio    │ (100 WAV files + metadata CSV)
│ Dataset      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Preprocessing│ 
│ - Trim to 30s│
│ - Format conv│
│ - Normalize  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Augmentation │ (300 additional samples)
│ - Add noise  │
│ - Pitch shift│
│ - Time stretch│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Feature      │ (142-dim vectors per sample)
│ Extraction   │
│ - MFCCs (13) │
│ - Mel-spec   │
│ - Roll-off   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Data Split   │ (80% train, 20% validation)
│ - Stratified │
│ - Random=42  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Model        │ (Random Forest)
│ Training     │
│ - 300 trees  │
│ - Max depth  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Hyperparameter│ (GridSearchCV)
│ Tuning       │
│ - 3-fold CV  │
│ - 27 combos  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Model        │ (65% accuracy)
│ Evaluation   │
│ - Confusion  │
│ - Metrics    │
└──────────────┘
```

## Memory and Performance Architecture

### Computational Complexity
- **Feature Extraction**: O(n × m) where n = samples, m = audio length
- **Model Training**: O(n × log(n) × d × t) where d = features, t = trees
- **Prediction**: O(log(n) × t) for inference

### Memory Usage
```
Component                Memory Usage
─────────────────────────────────────
Raw Audio (100 files)   ~500 MB
Processed Audio          ~500 MB
Augmented Audio          ~1.5 GB
Feature Matrix           ~0.5 MB (100×142 float64)
Trained Model            ~50 MB (300 trees)
Total Estimated          ~2.5 GB
```

## Error Handling Architecture

### Exception Management
```python
class AudioDeceptionPipeline:
    def __init__(self):
        self.error_handlers = {
            'file_not_found': self.handle_missing_file,
            'audio_corruption': self.handle_corrupt_audio,
            'feature_extraction_error': self.handle_feature_error,
            'model_training_error': self.handle_training_error
        }
    
    def process_with_error_handling(self, audio_path):
        try:
            # Process audio
        except Exception as e:
            # Route to appropriate handler
```

### Validation Checkpoints
1. **Input Validation**: File existence, format verification
2. **Processing Validation**: Audio loading success, duration check
3. **Feature Validation**: Feature vector completeness, NaN detection
4. **Model Validation**: Training success, convergence check

## Scalability Considerations

### Current Limitations
- **Dataset Size**: Limited to small datasets (100-1000 samples)
- **Real-time Processing**: Not optimized for streaming
- **Model Complexity**: Single model approach

### Potential Scaling Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Data Ingestion  │───▶│ Distributed      │───▶│ Model Serving   │
│ - Stream/Batch  │    │ Processing       │    │ - REST API      │
│ - Queue System  │    │ - Spark/Dask     │    │ - Load Balancer │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Technology Stack Architecture

### Core Dependencies
```
Application Layer:     Jupyter Notebook, Python 3.7+
ML Framework:         Scikit-learn 1.0+
Audio Processing:     Librosa 0.9+, PyDub 0.25+
Data Processing:      NumPy 1.21+, Pandas 1.3+
Visualization:        Matplotlib 3.5+, Seaborn 0.11+
File I/O:            SoundFile 0.10+
Optimization:        GridSearchCV (sklearn)
```

This architecture provides a comprehensive foundation for audio-based deception detection while maintaining modularity for future enhancements and scalability improvements.