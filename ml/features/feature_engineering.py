
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for fraud detection.

    The input DataFrame is not modified.
    """
    data = df.copy()

    # Transaction amount relative to the sender's previous balance
    data["amount_to_origin_balance_ratio"] = (
        data["amount"]
        / (data["oldbalanceOrg"] + 1)
    )

    # Transaction amount relative to the recipient's previous balance
    data["amount_to_destination_balance_ratio"] = (
        data["amount"]
        / (data["oldbalanceDest"] + 1)
    )

    # Difference between amount and recorded sender balance decrease
    data["origin_balance_error"] = (
        data["oldbalanceOrg"]
        - data["newbalanceOrig"]
        - data["amount"]
    )

    # Difference between amount and recorded recipient balance increase
    data["destination_balance_error"] = (
        data["newbalanceDest"]
        - data["oldbalanceDest"]
        - data["amount"]
    )

    # Transaction type as a categorical feature
    data["type"] = data["type"].astype("category")

    return data


if __name__ == "__main__":
    print("Feature engineering module loaded successfully.")