class FeaturePipeline:
    """Pipeline for computing task priority features."""

    def transform(self, df):
        """Add urgency and effort columns to the dataframe.

        urgency = 1 / days_left
        effort = difficulty * (1 - progress/100)
        """
        df["urgency"] = 1 / df["days_left"]
        df["effort"] = df["difficulty"] * (1 - df["progress"] / 100)
        return df
