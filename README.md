# Prepayment-Mortgage-Trading-Analysis-and-Prediction

Welcome to the Mortgage Trading Analysis and Prediction project! This repository contains the code and resources for a comprehensive case study on mortgage trading, designed to help you understand the financial system, sharpen your data modeling, DAX, and financial analysis skills, and experience the dynamic environment of a mortgage trading desk.

## Project Overview

As a newly hired junior trader on a trading desk for a mortgage originator, you will:
- Identify a population of mortgages to trade.
- Evaluate each mortgage and each bid received from prospective buyers.
- Execute trades based on your analysis.

This project aims to provide a unique perspective on the banking industry and improve your technical skills in data modeling, DAX, and financial analysis.

## Project Structure

The repository is organized as follows:

- `data/`: Contains the dataset used for analysis and prediction.
- `notebooks/`: Jupyter notebooks with step-by-step analysis and modeling.
- `scripts/`: Python scripts for data preprocessing, modeling, and evaluation.
- `reports/`: Generated reports and visualizations.
- `README.md`: This readme file.

## Dataset

The dataset used in this project is a synthetic mortgage dataset. It includes information on mortgage loans and their performance, such as loan amounts, interest rates, borrower credit scores, loan-to-value ratios, remaining terms, and whether the loan has defaulted.

**Dataset Information:**
- `loan_id`: Unique identifier for each loan
- `origination_date`: Date when the loan was originated
- `loan_amount`: Amount of the loan
- `interest_rate`: Interest rate of the loan
- `credit_score`: Borrower's credit score at the time of origination
- `loan_to_value`: Loan-to-value ratio at the time of origination
- `remaining_term`: Remaining term of the loan in months
- `monthly_payment`: Monthly payment amount
- `default`: Indicator of whether the loan has defaulted (binary: 0 for no default, 1 for default)
- `bid_price`: Price bid by prospective buyers (for simulation purposes)

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Technocolabs100/Mortgage-Prepayment-Analysis-and-Prediction.git
    cd Mortgage-Prepayment-Analysis-and-Prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preprocessing

Run the preprocessing script to clean and prepare the data:
```bash
python scripts/preprocess_data.py
```

### Exploratory Data Analysis (EDA)

Open the Jupyter notebooks in the `notebooks/` directory to perform exploratory data analysis:
```bash
jupyter notebook notebooks/eda.ipynb
```

### Modeling and Evaluation

Run the modeling script to train and evaluate the machine learning models:
```bash
python scripts/train_model.py
```

### Interactive Dashboard

To explore the interactive dashboard, use a BI tool like Power BI or Tableau and connect it to the processed data. The dashboard will help you visualize the mortgage portfolio, bids, and analysis results.

### Reporting

Generate reports and visualizations using the notebooks and scripts provided. The reports will summarize the trading activity, including the mortgages traded, bids received, and financial outcomes.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, please contact technocollabs@gmail.com

---

Enjoy the fast-paced and challenging environment of a mortgage trading desk as you delve into this comprehensive case study!
