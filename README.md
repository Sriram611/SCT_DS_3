# Bank Marketing Decision Tree Classifier

This project implements a **Decision Tree Classifier** to predict whether a customer will purchase a product or service based on their **demographic and behavioral data**.  
It is created as part of **Task 03** from a data science / machine learning assignment.

---

## Project Files

- `bank_marketing_sample.csv` – Sample dataset containing customer information  
- `decision_tree.py` or `decision_tree.ipynb` – Python code / Jupyter Notebook for training the model  
- `README.md` – Project documentation  

---

## Dataset Description

The dataset contains the following columns:

| Column | Description |
|--------|-------------|
| CustomerID | Unique ID for each customer |
| Age | Age of the customer |
| Job | Job type |
| Marital | Marital status |
| Education | Education level |
| Balance | Account balance |
| HousingLoan | Housing loan (yes/no) |
| PersonalLoan | Personal loan (yes/no) |
| Contact | Contact method |
| CampaignCalls | Number of marketing calls |
| Purchased | Target variable (0 = No, 1 = Yes) |

---

## Objective

To build a **Decision Tree Classifier** that predicts:
> **Whether a customer will purchase a product or service**

based on customer characteristics and past campaign behavior.

---

## How to Run the Project (Anaconda / Jupyter)

1. Install Anaconda from: https://www.anaconda.com  
2. Open **Anaconda Navigator**  
3. Launch **Jupyter Notebook** or **Spyder**  
4. Place the following files in the same folder:
   - `bank_marketing_sample.csv`
   - `decision_tree.ipynb` (or your Python file)
5. Run the program.

---

## Sample Code Used

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("bank_marketing_sample.csv")

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
```

---

## Output

- The model predicts whether a customer will **purchase (1)** or **not purchase (0)**.
- The final output displayed is the **accuracy of the model**.

---

## Conclusion

This project demonstrates:
- Data preprocessing using **one-hot encoding**
- Model training using a **Decision Tree Classifier**
- Evaluation using **accuracy**



