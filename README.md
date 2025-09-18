# German Credit Risk Analysis

This is a Streamlit web application for the Kaggle competition: [DSB-24 German Credit](https://www.kaggle.com/competitions/dsb-24-german-credit).

The application uses a **custom business-oriented metric** that considers the financial impact of loan decisions, taking into account loan amounts and real-world costs of false positives and false negatives in credit risk assessment.

## 📈 Custom Scoring Metric

The competition uses a specialized cost-sensitive metric that reflects real-world loan decision costs:

```python
def compute_costs(LoanAmount):
    return({
        'Risk_No Risk': 5.0 + .6 * LoanAmount,     # Cost of rejecting a good loan
        'No Risk_No Risk': 1.0 - .05 * LoanAmount, # Benefit of correctly rejecting
        'Risk_Risk': 1.0,                          # Cost of approving a risky loan
        'No Risk_Risk': 1.0                        # Benefit of correctly approving
    })

def score(solution, submission, row_id_column_name):
    # Adjusts for real-world class distribution (2% risky, 98% safe)
    real_prop = {'Risk': .02, 'No Risk': .98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}
    custom_weight = {
        'Risk': real_prop['Risk']/train_prop['Risk'], 
        'No Risk': real_prop['No Risk']/train_prop['No Risk']
    }
    
    costs = compute_costs(solution['LoanAmount'])
    # Calculate weighted cost based on prediction accuracy and loan amounts
    loss = (y_true=='Risk') * custom_weight['Risk'] * \
           ((y_pred=='Risk') * costs['Risk_Risk'] + (y_pred=='No Risk') * costs['Risk_No Risk']) + \
           (y_true=='No Risk') * custom_weight['No Risk'] * \
           ((y_pred=='Risk') * costs['No Risk_Risk'] + (y_pred=='No Risk') * costs['No Risk_No Risk'])
    
    return loss.mean()
```

**Key Features:**
- **Loan Amount Dependency**: Costs scale with loan size (larger loans = higher rejection cost)
- **Real-World Distribution**: Adjusts for actual 2% default rate vs 33% in training data
- **Business Impact**: Minimizes total financial loss rather than maximizing accuracy

## 🎯 Features

- **Interactive Model Comparison**: Compare XGBoost and SVM models side-by-side
- **One-Click Kaggle Predictions**: Generate predictions for the competition test data instantly
- **Custom Leaderboard Metric**: Business-focused scoring that reflects real loan costs
- **Feature Importance Analysis**: Understand which factors drive credit decisions
- **Comprehensive Evaluation**: Multiple metrics including precision, recall, F1-score, and custom business score

## 🚀 Quick Start

**Requirements**: Python 3.10.*

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** to `http://localhost:8501`

4. **Generate Kaggle predictions**: Click one button to predict the predefined test data from the Kaggle competition and download results!

## 📊 Models

- **XGBoost**: Gradient boosting classifier optimized for the custom metric
- **Support Vector Machine**: SVM with RBF kernel for comparison

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

## 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── streamlit_components/     # UI components
├── src/                      # Core logic
│   ├── models/              # Model training and evaluation
│   ├── tools/               # Custom metrics and utilities
│   └── utils/               # Data processing utilities
├── tests/                   # Unit / Integration tests
├── data/                    # Dataset files
└── requirements.txt         # Dependencies
```

## 🏆 Competition Context

This project addresses the German Credit Risk dataset with a focus on **business value** rather than just statistical accuracy. The custom metric weighs misclassification costs based on loan amounts, making it more aligned with real-world financial decision-making.

---

*Built with ❤️ using Streamlit, XGBoost, and scikit-learn*