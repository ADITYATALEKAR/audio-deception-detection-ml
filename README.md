# Audio-Based Deception Detection System

A machine learning system for detecting deception in audio recordings using advanced feature extraction and Random Forest classification. This project analyzes 30-second narrated stories to classify them as truthful or deceptive based on acoustic patterns.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Overview

This project implements a machine learning pipeline for detecting deception in speech recordings. By analyzing acoustic features such as pitch variations, frequency patterns, and spectral characteristics, the system can classify audio narratives as either truthful or deceptive with 65% accuracy.

### Key Objectives
- Build a robust audio preprocessing pipeline
- Extract meaningful acoustic features from speech recordings
- Develop and optimize machine learning models for binary classification
- Evaluate model performance and identify areas for improvement

## Features

- **Audio Preprocessing**: Automated trimming to standardize 30-second duration
- **Feature Extraction**: Advanced acoustic feature extraction using Librosa
  - MFCCs (Mel Frequency Cepstral Coefficients)
  - Mel-spectrograms
  - Spectral roll-off
- **Data Augmentation**: Multiple augmentation techniques to enhance dataset diversity
  - Noise addition
  - Pitch shifting
  - Time stretching
- **Model Optimization**: Hyperparameter tuning using GridSearchCV
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

## Dataset

The project uses the MLEnd Deception Dataset containing:
- **100 audio samples**: 30-second narrated stories
- **Balanced distribution**: 50 truthful stories, 50 deceptive stories
- **Metadata**: Language information and story classifications

**Dataset Sources:**
- Audio recordings: [MLEndDD stories small](https://github.com/MLEndDatasets/Deception/tree/main/MLEndDD_stories_small)
- Metadata: [Story attributes small](https://github.com/MLEndDatasets/Deception/blob/main/MLEndDD_story_attributes_small.csv)

## Installation

### Prerequisites
- Python 3.7+
- Required libraries (see requirements.txt)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-deception-detection-ml.git
cd audio-deception-detection-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download audio files from the MLEnd repository
   - Place audio files in `data/audio/` directory
   - Place metadata CSV in `data/` directory

### Required Libraries
```
librosa>=0.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pydub>=0.25.0
soundfile>=0.10.0
```

## Usage

### Basic Usage
1. **Data Preprocessing**:
```python
# Trim audio files to 30 seconds
python src/preprocessing.py --input_dir data/audio/ --output_dir data/processed/
```

2. **Feature Extraction**:
```python
# Extract acoustic features
python src/feature_extraction.py --audio_dir data/processed/ --output features.npy
```

3. **Model Training**:
```python
# Train and evaluate the model
python src/train_model.py --features features.npy --metadata data/metadata.csv
```

### Running the Complete Pipeline
Execute the Jupyter notebook:
```bash
jupyter notebook ECS7020P_miniproject_2425.ipynb
```

## Methodology

### 1. Data Preprocessing
- **Audio Trimming**: Standardize all recordings to 30-second duration
- **Format Conversion**: Convert to WAV format for consistent processing

### 2. Feature Engineering
Extract multiple acoustic features to capture speech characteristics:

**MFCCs (Mel Frequency Cepstral Coefficients)**
- Capture frequency spectrum characteristics
- 13 coefficients extracted per audio sample

**Mel-spectrograms**
- Time-frequency representation in Mel scale
- Captures energy distribution across frequency bands

**Spectral Roll-off**
- Frequency below which specified percentage of energy is concentrated
- Indicates tonality and energy distribution

### 3. Data Augmentation
Enhance dataset diversity with three augmentation techniques:
- **Noise Addition**: Simulate real-world recording conditions
- **Pitch Shifting**: Account for speaker voice variations
- **Time Stretching**: Handle different speaking speeds

### 4. Model Development
**Base Model**: Random Forest Classifier
- Chosen for robustness with tabular data
- Handles non-linear relationships effectively
- Provides feature importance insights

**Hyperparameter Tuning**: GridSearchCV optimization
- n_estimators: [100, 200, 300]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5, 10]

### 5. Evaluation Metrics
- Accuracy score
- Precision, Recall, F1-score for both classes
- Confusion matrix visualization
- Cross-validation performance

## Results

### Model Performance
- **Validation Accuracy**: 65%
- **Improvement over Baseline**: +15% (from 50% baseline)

### Detailed Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Deceptive (0) | 1.00 | 0.50 | 0.67 | 14 |
| True (1) | 0.46 | 1.00 | 0.63 | 6 |
| **Accuracy** | | | **0.65** | **20** |
| **Macro Avg** | 0.73 | 0.75 | 0.65 | 20 |
| **Weighted Avg** | 0.84 | 0.65 | 0.66 | 20 |

### Key Findings
- **Strengths**: High precision for deceptive stories (100%), perfect recall for true stories
- **Challenges**: Lower recall for deceptive stories, precision issues with true story classification
- **Optimal Parameters**: n_estimators=300, max_depth=10, min_samples_split=2

## Project Structure

```
audio-deception-detection-ml/
│
├── data/
│   ├── audio/                 # Original audio files
│   ├── processed/            # Preprocessed audio files
│   └── metadata.csv          # Dataset metadata
│
├── src/
│   ├── preprocessing.py      # Audio preprocessing functions
│   ├── feature_extraction.py # Feature extraction utilities
│   ├── augmentation.py       # Data augmentation techniques
│   └── train_model.py        # Model training and evaluation
│
├── notebooks/
│   └── ECS7020P_miniproject_2425.ipynb  # Main analysis notebook
│
├── results/
│   ├── confusion_matrix.png  # Performance visualizations
│   └── feature_importance.png
│
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── LICENSE                  # License file
```

## Future Work

### Immediate Improvements
1. **Dataset Expansion**: Increase sample size and diversity
2. **Feature Engineering**: Explore additional acoustic features
   - Chroma features
   - Spectral contrast
   - Zero crossing rate
3. **Advanced Models**: Experiment with XGBoost, Neural Networks
4. **Class Balancing**: Address imbalanced performance between classes

### Long-term Enhancements
1. **Real-time Detection**: Implement streaming audio analysis
2. **Multi-language Support**: Extend to diverse linguistic contexts
3. **Deep Learning**: Explore CNN/RNN architectures for audio
4. **Ensemble Methods**: Combine multiple model predictions

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

### Dataset
- MLEnd Deception Dataset: [GitHub Repository](https://github.com/MLEndDatasets/Deception)

### Research Papers
1. Zhang, S. et al. (2023). "Navigating the Soundscape of Deception: A Comprehensive Survey on Audio Deepfake Detection." IEEE.
2. Kumar, A. et al. (2023). "Unmasking Audio Deception: Performance Analysis in Machine Learning-Based Detection." IEEE.
3. Bahaa, M. et al. (2024). "Advancing Automated Deception Detection: A Multimodal Approach to Feature Extraction and Analysis." arXiv.

### Libraries and Tools
- [Librosa](https://librosa.org/) - Audio analysis library
- [Scikit-learn](https://scikit-learn.org/) - Machine learning framework
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation

---

## Acknowledgments

- MLEnd Dataset contributors for providing the deception detection dataset
- Course instructors and peers for guidance and feedback
- Open-source community for the excellent libraries and tools
