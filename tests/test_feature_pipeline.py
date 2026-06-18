import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from feature_pipeline import FeaturePipeline


class TestFeaturePipeline:
    """Tests for the FeaturePipeline class."""

    def setup_method(self):
        self.pipeline = FeaturePipeline()

    def test_transform_adds_urgency_column(self):
        """Transform should add an 'urgency' column."""
        df = pd.DataFrame({
            "days_left": [10, 5, 2],
            "difficulty": [3, 7, 5],
            "progress": [50, 20, 80]
        })
        result = self.pipeline.transform(df)
        assert "urgency" in result.columns

    def test_transform_adds_effort_column(self):
        """Transform should add an 'effort' column."""
        df = pd.DataFrame({
            "days_left": [10, 5, 2],
            "difficulty": [3, 7, 5],
            "progress": [50, 20, 80]
        })
        result = self.pipeline.transform(df)
        assert "effort" in result.columns

    def test_urgency_calculation(self):
        """Urgency should be 1 / days_left."""
        df = pd.DataFrame({
            "days_left": [10, 5, 2, 1],
            "difficulty": [1, 1, 1, 1],
            "progress": [0, 0, 0, 0]
        })
        result = self.pipeline.transform(df)
        expected = [0.1, 0.2, 0.5, 1.0]
        np.testing.assert_array_almost_equal(result["urgency"].values, expected)

    def test_effort_calculation(self):
        """Effort should be difficulty * (1 - progress/100)."""
        df = pd.DataFrame({
            "days_left": [10, 10, 10, 10],
            "difficulty": [5, 10, 8, 3],
            "progress": [0, 50, 75, 100]
        })
        result = self.pipeline.transform(df)
        expected = [5.0, 5.0, 2.0, 0.0]
        np.testing.assert_array_almost_equal(result["effort"].values, expected)

    def test_transform_returns_dataframe(self):
        """Transform should return a pandas DataFrame."""
        df = pd.DataFrame({
            "days_left": [5],
            "difficulty": [3],
            "progress": [40]
        })
        result = self.pipeline.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_preserves_original_columns(self):
        """Transform should keep existing columns intact."""
        df = pd.DataFrame({
            "days_left": [5],
            "difficulty": [3],
            "progress": [40],
            "task_name": ["Fix bug"]
        })
        result = self.pipeline.transform(df)
        assert "days_left" in result.columns
        assert "difficulty" in result.columns
        assert "progress" in result.columns
        assert "task_name" in result.columns

    def test_effort_zero_when_complete(self):
        """Effort should be 0 when progress is 100%."""
        df = pd.DataFrame({
            "days_left": [5],
            "difficulty": [10],
            "progress": [100]
        })
        result = self.pipeline.transform(df)
        assert result["effort"].iloc[0] == 0.0

    def test_effort_equals_difficulty_when_no_progress(self):
        """Effort should equal difficulty when progress is 0%."""
        df = pd.DataFrame({
            "days_left": [5],
            "difficulty": [7],
            "progress": [0]
        })
        result = self.pipeline.transform(df)
        assert result["effort"].iloc[0] == 7.0

    def test_urgency_increases_as_deadline_approaches(self):
        """Urgency should increase as days_left decreases."""
        df = pd.DataFrame({
            "days_left": [10, 5, 1],
            "difficulty": [1, 1, 1],
            "progress": [0, 0, 0]
        })
        result = self.pipeline.transform(df)
        urgency_values = result["urgency"].values
        assert urgency_values[0] < urgency_values[1] < urgency_values[2]

    def test_transform_with_float_values(self):
        """Should handle float values correctly."""
        df = pd.DataFrame({
            "days_left": [2.5],
            "difficulty": [4.5],
            "progress": [33.3]
        })
        result = self.pipeline.transform(df)
        assert np.isclose(result["urgency"].iloc[0], 1 / 2.5)
        assert np.isclose(result["effort"].iloc[0], 4.5 * (1 - 33.3 / 100))

    def test_transform_multiple_rows(self):
        """Should handle multiple rows correctly."""
        df = pd.DataFrame({
            "days_left": [1, 2, 3, 4, 5],
            "difficulty": [10, 8, 6, 4, 2],
            "progress": [0, 25, 50, 75, 100]
        })
        result = self.pipeline.transform(df)
        assert len(result) == 5
        assert result["urgency"].iloc[0] == 1.0
        assert result["effort"].iloc[-1] == 0.0

    def test_transform_modifies_in_place(self):
        """Transform modifies the original dataframe (in-place behavior)."""
        df = pd.DataFrame({
            "days_left": [5],
            "difficulty": [3],
            "progress": [50]
        })
        result = self.pipeline.transform(df)
        # The function modifies df in place and also returns it
        assert "urgency" in df.columns
        assert "effort" in df.columns
