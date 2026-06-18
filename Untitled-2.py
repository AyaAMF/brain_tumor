class FeaturePipeline:

    def transform(self, df):
        df["urgency"] = df["days_left"].apply(lambda d: 1 / d if d != 0 else float("inf"))
        df["effort"] = df["difficulty"] * (1 - df["progress"]/100)
        return df