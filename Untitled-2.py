class FeaturePipeline:

    def transform(self, df):
        df["urgency"] = 1 / df["days_left"]
        df["effort"] = df["difficulty"] * (1 - df["progress"]/100)
        return df