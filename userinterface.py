from pathlib import Path
import sys
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "Mental_Health_Model.pkl"

# These columns and their order must match the notebook's feature_col.
FEATURE_COLUMNS = [
    "Study_Hours",
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_country",
]

STRESS_LEVELS = ["Low", "Medium", "High", "Very High"]


def ask_number(label, minimum=None, maximum=None, integer=False):
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = int(raw) if integer else float(raw)
            if minimum is not None and value < minimum:
                print(f"Please enter a value >= {minimum}.")
                continue
            if maximum is not None and value > maximum:
                print(f"Please enter a value <= {maximum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def ask_choice(label, choices):
    print(f"\n{label}")
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}. {choice}")

    while True:
        raw = input("Enter option number or type the value: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        for choice in choices:
            if raw.casefold() == choice.casefold():
                return choice
        print("Invalid choice. Select a listed option or type its exact value.")


def collect_input():
    print("\n=== Student Mental Health Score Prediction ===")
    print("Enter the student's details. Use hours for time-related fields.\n")

    data = {
         "Age": ask_number("Age in years", 1, 100, integer=True),
        "Gender": input("Gender (Male/Female): ").strip(),
        "Academic_Level": input("Academic level (Graduate/ Undergraduate/High School): ").strip(),
        "Study_Hours": ask_number("Daily study hours", 0, 24),
        "Avg_Daily_Usage_Hours": ask_number("Average daily social-media usage (hours)", 0, 24),
        "Daily_Unlocks": ask_number("Average phone unlocks per day", 0, 10000, integer=True),
        "Most_Used_Platform": input("Most-used social media platform: ").strip(),
        "Purpose_Of_Use": input("Main purpose of social media use: ").strip(),
        "Physical_Activity_Hours": ask_number("Daily physical activity (hours)", 0, 24),
        "Sleep_Hours_Per_Night": ask_number("Sleep per night (hours)", 0, 24),
        "Stress_Level": ask_choice("Stress level", STRESS_LEVELS),
    }

    country = input( "enter country name: ").strip()
    data["Grouped_country"] = country if country else "Other"

    # Reject blank categorical inputs rather than silently passing bad data.
    for column in [
        "Gender", "Academic_Level", "Most_Used_Platform",
        "Purpose_Of_Use", "Grouped_country"
    ]:
        if not data[column]:
            raise ValueError(f"{column} cannot be blank.")

    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def main():
    if not MODEL_PATH.exists():
        print(f"\nModel file not found: {MODEL_PATH}")
        print("First run the training notebook and save the fitted pipeline as "
              "'Mental_Health_Model.pkl'.")
        sys.exit(1)

    try:
        model = joblib.load(MODEL_PATH)
        input_df = collect_input()
        prediction = float(model.predict(input_df)[0])

        print("\n" + "=" * 48)
        print(f"Predicted Mental Health Score: {prediction:.2f}")
        print("=" * 48)
        print("This is a model estimate based on the training dataset, not a "
              "medical or psychological diagnosis.")
    except (ValueError, TypeError) as exc:
        print(f"\nInput error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nPrediction failed: {exc}")
        print("Check that the model was saved from the same feature pipeline "
              "and that your scikit-learn version matches the training version.")
        sys.exit(1)


if __name__ == "__main__":
    main()
