from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_PATIENTS = 2000

OUTPUT_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "generated"
    / "synthetic_readmission_data.csv"
)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    age = rng.integers(
        40,
        91,
        NUMBER_OF_PATIENTS,
    )

    prior_admissions = np.clip(
        rng.poisson(
            lam=1.2,
            size=NUMBER_OF_PATIENTS,
        ),
        0,
        6,
    )

    length_of_stay = np.clip(
        rng.gamma(
            shape=2.0,
            scale=2.5,
            size=NUMBER_OF_PATIENTS,
        ),
        1,
        20,
    )

    chronic_condition_count = np.clip(
        rng.poisson(
            lam=2.5,
            size=NUMBER_OF_PATIENTS,
        ),
        0,
        8,
    )

    medication_count = np.clip(
        (
            chronic_condition_count * 2
            + rng.normal(
                2,
                2,
                NUMBER_OF_PATIENTS,
            )
        ),
        0,
        20,
    )

    recent_ed_visit = rng.binomial(
        1,
        0.25,
        NUMBER_OF_PATIENTS,
    )

    hba1c = np.clip(
        rng.normal(
            7.1,
            1.4,
            NUMBER_OF_PATIENTS,
        ),
        4.5,
        13.0,
    )

    systolic_bp = np.clip(
        rng.normal(
            132,
            18,
            NUMBER_OF_PATIENTS,
        ),
        90,
        200,
    )

    egfr = np.clip(
        rng.normal(
            72,
            25,
            NUMBER_OF_PATIENTS,
        ),
        10,
        130,
    )

    followup_days = rng.integers(
        1,
        31,
        NUMBER_OF_PATIENTS,
    )

    # --------------------------------------------------
    # Synthetic risk-generation formula
    #
    # This creates a learnable relationship for our
    # educational ML demonstration.
    #
    # It is NOT a clinical risk equation.
    # --------------------------------------------------

    risk_logit = (
        -5.0
        + 0.018 * (age - 60)
        + 0.55 * prior_admissions
        + 0.07 * length_of_stay
        + 0.32 * chronic_condition_count
        + 0.06 * medication_count
        + 0.85 * recent_ed_visit
        + 0.16 * (hba1c - 7)
        + 0.008 * (systolic_bp - 130)
        - 0.018 * (egfr - 70)
        + 0.035 * (followup_days - 7)
    )

    risk_probability = sigmoid(
        risk_logit
    )

    readmitted_30d = rng.binomial(
        1,
        risk_probability,
    )

    dataframe = pd.DataFrame(
        {
            "age": age,
            "prior_admissions_12m": (
                prior_admissions
            ),
            "length_of_stay_days": (
                np.round(
                    length_of_stay,
                    1,
                )
            ),
            "chronic_condition_count": (
                chronic_condition_count
            ),
            "medication_count": (
                np.round(
                    medication_count,
                ).astype(int)
            ),
            "recent_ed_visit": (
                recent_ed_visit
            ),
            "hba1c": np.round(
                hba1c,
                1,
            ),
            "systolic_bp": np.round(
                systolic_bp,
            ).astype(int),
            "egfr": np.round(
                egfr,
                1,
            ),
            "followup_days": (
                followup_days
            ),
            "readmitted_30d": (
                readmitted_30d
            ),
        }
    )

    return dataframe


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = generate_dataset()

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    readmission_rate = (
        dataframe[
            "readmitted_30d"
        ].mean()
    )

    print(
        f"Generated {len(dataframe)} "
        "synthetic patient records"
    )

    print(
        f"Readmission rate: "
        f"{readmission_rate:.2%}"
    )

    print(
        f"Saved dataset to: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("First five records:")
    print(
        dataframe.head().to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()