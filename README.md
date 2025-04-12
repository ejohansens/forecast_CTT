# Forecast MVP for CTT

Data extraction from Postgresql database, normalization, encoding, training, prediction making and statistics.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ejohansens/forecast_CTT.git
    ```
2. Create a virtual environment (optional):
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Set up the database using Postgresql (or optionally, ask the authors for the whole dataset file and skip this step. You will need to comment out the code and go straight to the data preprocessing part.)
2. Change the database credentials in extractTrain.py and extractValidation.py
3. Run the python files in the following order: extractTrain.py > extractValidation.py > predict.py
4. After predict.py is run, it will provide you with some charts, statistics in the console and three files, most importantly converted_adjusted_predicted_bookings.json which will have the predicted times and booking_id. adjusted_predicted_bookings.json is the dataset which was used for validation and comparison_results.json provides comparison between actual and predicted times.
