# NBA Draft Prediction

This project aims to predict whether a college basketball player will be drafted into the NBA based on their season statistics. It utilizes advanced machine learning algorithms to analyze college player statistics and predict their likelihood of being drafted into the NBA.

## Table of Contents

- [NBA Draft Prediction](#nba-draft-prediction)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

The NBA draft is an annual event where teams select players from American colleges and international professional leagues. This project uses a dataset of college basketball players' statistics to predict their likelihood of being drafted into the NBA. The primary metric for model evaluation is the Area Under the Receiver Operating Characteristic Curve (AUROC).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/vishalraj247/NBA-Draft-Prediction.git
   ```

2. Navigate to the project directory:
   ```
   cd NBA-Draft-Prediction
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Data Preparation:
   - Ensure the data is in the `data/` directory.
   - Import the datasets:
     ```
     python src/data/make_dataset.py
     ```
   - Run the data preprocessing script:
     ```
     python src/data/preprocess_data.py
     ```

2. Feature Selection:
   - Select the features using:
     ```
     python src/features/build_features.py
     ```

3. Model Training:
   - Train the model using:
     ```
     python src/models/train_model.py
     ```

4. Prediction:
   - Use the trained model to make predictions:
     ```
     python src/models/predict.py
     ```

## Contributing

1. Fork the project.
2. Create your feature branch: `git checkout -b feature/YourFeatureName`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeatureName`
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
