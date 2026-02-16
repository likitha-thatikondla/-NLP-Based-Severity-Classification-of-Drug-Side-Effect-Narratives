# Drug Side-Effect Severity Classification System

An NLP-based decision support system that analyzes patient-reported drug side effects and provides automated severity classification with medical recommendations.

## 🎯 Problem Statement

Hospitals and pharmaceutical companies receive thousands of adverse drug reaction reports daily. Manual review is time-consuming and risks missing critical cases. This system automates the triage process, instantly identifying which cases need immediate medical attention.

## 🚀 Solution

A machine learning pipeline that:
- Reads free-text symptom descriptions
- Predicts severity level (0-4 scale)
- Detects unexpected side effects
- Provides actionable medical recommendations

## 📊 System Architecture

```
Input (Drug + Symptoms)
    ↓
Text Preprocessing & Cleaning
    ↓
Drug Profile Matching (Expected vs Unexpected)
    ↓
TF-IDF Vectorization
    ↓
ML Classification (Logistic Regression)
    ↓
Decision Logic + Safety Rules
    ↓
Output (Severity + Recommendation)
```

## 📁 Project Structure

```
├── 01_build_drug_profiles.ipynb    # Creates drug knowledge base
├── 02_severity_model.ipynb         # Trains ML classifier
├── 03_decision_engine.ipynb        # Production system
├── drugLibTrain_raw.tsv            # Training data
├── drugLibTest_raw.tsv             # Test data
└── models/                          # Saved models
    ├── drug_profiles.pkl
    ├── severity_model.pkl
    └── tfidf_vectorizer.pkl
```

## 🔧 Installation

```bash
pip install pandas numpy scikit-learn joblib jupyter
```

## 📖 Usage

### 1. Build Drug Profiles
```bash
jupyter notebook 01_build_drug_profiles.ipynb
# Run all cells
```

### 2. Train Severity Model
```bash
jupyter notebook 02_severity_model.ipynb
# Run all cells
```

### 3. Run Decision Engine
```bash
jupyter notebook 03_decision_engine.ipynb
# Run all cells
```

### Quick Prediction

```python
from models import drug_decision_support

result = drug_decision_support("lyrica", "severe chest pain")

print(f"Severity: {result['Severity']}")
print(f"Confidence: {result['Confidence']}")
print(f"Action: {result['Recommendation']}")
```

**Output:**
```
Severity: Extremely Severe
Confidence: 92.3%
Action: 🚨 EMERGENCY - Seek Medical Attention Immediately
```

## 🧠 Technical Approach

### Key Components

**1. Drug Profiles (Notebook 1)**
- Uses TF-IDF to extract common side effects for each drug
- Builds knowledge base of expected vs unexpected symptoms
- Stores severity distribution for 99 drugs

**2. Severity Classifier (Notebook 2)**
- Multi-class classification (5 severity levels)
- Logistic Regression with balanced class weights
- TF-IDF features with bigrams (5000 features)
- Handles class imbalance in training data

**3. Decision Engine (Notebook 3)**
- Combines drug profiles + ML predictions
- Rule-based safety checks for critical keywords
- Confidence-based recommendation thresholds
- Clean output format for production use

### ML Techniques Used

| Technique | Purpose |
|-----------|---------|
| TF-IDF | Text vectorization |
| N-grams (1,2) | Capture medical phrases |
| Logistic Regression | Multi-class classification |
| Class Balancing | Handle imbalanced data |
| Probability Calibration | Accurate confidence scores |

## 📈 Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 85% |
| Macro F1-Score | 0.87 |
| Weighted F1-Score | 0.85 |
| Inference Time | <10ms |

### Class-Specific Performance

| Severity Level | F1-Score |
|----------------|----------|
| No Side Effects | 0.89 |
| Mild | 0.83 |
| Moderate | 0.82 |
| Severe | 0.88 |
| Extremely Severe | 0.91 |

## 🔍 Key Features

✅ **Fast**: Predictions in <10ms  
✅ **Accurate**: 85% overall accuracy, 91% F1 on critical cases  
✅ **Safe**: Multiple fail-safes for medical safety  
✅ **Explainable**: Shows matched symptoms and confidence  
✅ **Scalable**: Handles 1000+ predictions/second  

## 🛡️ Safety Mechanisms

1. **Critical Keyword Detection**: Hardcoded rules for emergency keywords (chest pain, breathing, etc.)
2. **Confidence Thresholding**: Low confidence → escalate to doctor
3. **Expectation Matching**: Flags unexpected side effects for review
4. **Default to Caution**: When uncertain, system recommends medical attention

## 🎯 Use Cases

- **Hospital Triage**: Automatically prioritize incoming adverse event reports
- **Pharmacovigilance**: Post-market drug safety monitoring
- **Telemedicine**: Virtual consultation support
- **Clinical Trials**: Real-time safety signal detection

## 📊 Sample Outputs

### Example 1: Expected Mild Symptoms
```
Input: Drug="Aspirin", Symptoms="mild stomach discomfort"
Output: 
  Severity: Mild (65.2%)
  → ✓ Mild - Continue Monitoring
```

### Example 2: Unexpected Severe Symptoms
```
Input: Drug="Lyrica", Symptoms="severe chest pain and fainting"
Output:
  Severity: Extremely Severe (92.3%)
  → 🚨 EMERGENCY - Seek Medical Attention Immediately
```

### Example 3: Moderate Symptoms
```
Input: Drug="Xanax", Symptoms="extreme drowsiness and confusion"
Output:
  Severity: Moderate (78.1%)
  → ℹ️ Monitor Symptoms - Contact Doctor if Persists
```

## 🔄 Workflow

1. **User Input**: Drug name + symptom description
2. **Preprocessing**: Text cleaning, normalization
3. **Feature Extraction**: TF-IDF vectorization
4. **Drug Matching**: Check against known side effects
5. **Severity Prediction**: ML model classification
6. **Decision Logic**: Apply safety rules + thresholds
7. **Output**: Severity + confidence + recommendation

## 🚀 Future Enhancements

- [ ] Deep learning with BioBERT for better context understanding
- [ ] Negation handling ("no pain" vs "pain")
- [ ] Multi-modal input (vitals, images)
- [ ] Patient history integration
- [ ] Real-time model updates (online learning)
- [ ] Explainability dashboard (SHAP/LIME)

## 📝 Technical Details

**Languages**: Python 3.9+  
**Libraries**: scikit-learn, pandas, numpy, joblib  
**ML Algorithm**: Logistic Regression with L2 regularization  
**Feature Engineering**: TF-IDF with bigrams, stop word removal  
**Model Size**: ~1MB (lightweight)  
**Deployment**: CPU-only, no GPU required  

## ⚖️ Limitations

- Does not replace professional medical advice
- Limited to 99 drugs in current knowledge base
- Cannot handle negation well ("no pain")
- Requires English text input
- Trained on user-reported data (not clinical trials)

## 🤝 Contributing

This is an academic/portfolio project. For production use:
- Validate on clinical data
- Obtain medical professional review
- Ensure regulatory compliance (FDA, HIPAA)
- Implement proper audit trails

---

**Status**: ✅ Complete | **Version**: 1.0.0 | **Last Updated**: February 2026
