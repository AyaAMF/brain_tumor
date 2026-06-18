import logging

import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"days_left", "difficulty", "progress"}


class FeaturePipeline:

    def transform(self, df):
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        if (df["days_left"] == 0).any():
            logger.warning(
                "days_left contains zeros; urgency will be inf for those rows"
            )
        df["urgency"] = np.where(
            df["days_left"] == 0, np.inf, 1 / df["days_left"]
        )

        df["effort"] = df["difficulty"] * (1 - df["progress"] / 100)
        return df