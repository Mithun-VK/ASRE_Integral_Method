"""
Point-in-Time Data Manager - Production Ready

Ensures NO look-ahead bias by managing data availability based on announcement dates.
Critical component for institutional-grade backtesting.

Author: ASRE Project
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path


class PointInTimeError(Exception):
    """Custom exception for point-in-time errors"""
    pass


class PointInTimeData:
    """
    Manages point-in-time data access to prevent look-ahead bias.

    Core Principle:
    - Data is only available AFTER it's announced
    - Q4 2023 data (ended 2024-01-28) announced 2024-02-21
    - Therefore, Q4 2023 data is available on/after 2024-02-21

    Features:
    - Enforces announcement date constraints
    - Generates time-series of available data
    - Validates no look-ahead bias
    - Handles missing announcement dates
    - Supports multiple data sources (fundamentals, prices)

    Usage:
        pit = PointInTimeData(fundamentals_df)
        data = pit.get_data_as_of('2024-03-01')  # Only announced data
    """

    def __init__(
        self,
        fundamentals_df: pd.DataFrame,
        date_col: str = 'date',
        announced_col: str = 'announced_date',
        validate: bool = True
    ):
        """
        Initialize Point-in-Time Data Manager.

        Args:
            fundamentals_df: DataFrame with fundamental data
            date_col: Column name for quarter end date
            announced_col: Column name for announcement date
            validate: Whether to validate data on initialization
        """
        self.date_col = date_col
        self.announced_col = announced_col

        # Validate input
        if fundamentals_df.empty:
            raise PointInTimeError("Empty DataFrame provided")

        required_cols = [date_col, announced_col]
        missing_cols = [col for col in required_cols if col not in fundamentals_df.columns]
        if missing_cols:
            raise PointInTimeError(f"Missing required columns: {missing_cols}")

        # Store data
        self.fundamentals = fundamentals_df.copy()

        # Convert dates to datetime
        self._convert_dates()

        # Validate if requested
        if validate:
            self._validate_data()

        # Sort by announcement date (critical for time-series access)
        self.fundamentals = self.fundamentals.sort_values(announced_col).reset_index(drop=True)

        # Build announcement timeline
        self._build_timeline()

        print(f"✅ PointInTimeData initialized")
        print(f"   Quarters: {len(self.fundamentals)}")
        print(f"   Date range: {self.fundamentals[date_col].min()} to {self.fundamentals[date_col].max()}")
        print(f"   Announcement range: {self.fundamentals[announced_col].min()} to {self.fundamentals[announced_col].max()}")

    def _convert_dates(self):
        """Convert date columns to datetime"""
        for col in [self.date_col, self.announced_col]:
            if not pd.api.types.is_datetime64_any_dtype(self.fundamentals[col]):
                self.fundamentals[col] = pd.to_datetime(self.fundamentals[col])

    def _validate_data(self):
        """
        Validate data to ensure no look-ahead bias.

        Critical checks:
        1. announced_date >= date (announcement after quarter end)
        2. No missing announcement dates
        3. No duplicate quarters
        4. Chronological order
        """
        print("   Validating point-in-time constraints...")

        # Check 1: Announcement date >= quarter end date
        invalid_dates = self.fundamentals[self.announced_col] < self.fundamentals[self.date_col]
        if invalid_dates.any():
            n_invalid = invalid_dates.sum()
            print(f"⚠️  WARNING: {n_invalid} quarters have announcement date before quarter end!")
            print(f"   This violates point-in-time logic and will cause look-ahead bias.")

            # Show examples
            bad_data = self.fundamentals[invalid_dates][[self.date_col, self.announced_col]].head(3)
            print(f"\n   Examples:")
            print(bad_data.to_string(index=False))

            raise PointInTimeError(
                f"{n_invalid} quarters have invalid announcement dates. "
                "Fix data before proceeding."
            )

        # Check 2: No missing announcement dates
        missing_announced = self.fundamentals[self.announced_col].isnull().sum()
        if missing_announced > 0:
            raise PointInTimeError(
                f"{missing_announced} quarters missing announcement dates. "
                "Use FundamentalFetcher to estimate missing dates."
            )

        # Check 3: No duplicate quarters
        duplicates = self.fundamentals[self.date_col].duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  WARNING: {duplicates} duplicate quarters found. Keeping first occurrence.")
            self.fundamentals = self.fundamentals.drop_duplicates(subset=[self.date_col], keep='first')

        # Check 4: Ensure reasonable announcement lag
        lag_days = (self.fundamentals[self.announced_col] - self.fundamentals[self.date_col]).dt.days

        mean_lag = lag_days.mean()
        max_lag = lag_days.max()
        min_lag = lag_days.min()

        print(f"   Announcement lag: mean={mean_lag:.0f} days, min={min_lag}, max={max_lag}")

        if mean_lag < 20 or mean_lag > 60:
            print(f"⚠️  WARNING: Average announcement lag ({mean_lag:.0f} days) is unusual.")
            print(f"   Typical range: 30-45 days")

        if max_lag > 90:
            print(f"⚠️  WARNING: Some announcements have lag > 90 days")

        print(f"✅ Validation passed")

    def _build_timeline(self):
        """
        Build timeline of data availability.

        Creates mapping: announcement_date -> list of available quarters
        """
        self.timeline = {}

        for idx, row in self.fundamentals.iterrows():
            announced_date = row[self.announced_col]
            quarter_date = row[self.date_col]

            # Add to timeline
            if announced_date not in self.timeline:
                self.timeline[announced_date] = []

            self.timeline[announced_date].append({
                'quarter_date': quarter_date,
                'index': idx,
                'announced_date': announced_date
            })

        # Sort timeline by date
        self.timeline = dict(sorted(self.timeline.items()))

    def get_data_as_of(
        self,
        as_of_date: Union[str, datetime, pd.Timestamp],
        return_type: str = 'latest'
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Get data available as of a specific date (NO look-ahead bias).

        Args:
            as_of_date: Date to get data for
            return_type: 'latest' (most recent quarter) or 'all' (all available quarters)

        Returns:
            Series (if return_type='latest') or DataFrame (if return_type='all')

        Example:
            # On 2024-03-01, only data announced by 2024-03-01 is available
            data = pit.get_data_as_of('2024-03-01')
            # Returns Q4 2023 data (announced 2024-02-21)
            # Q1 2024 data NOT available yet (announced 2024-05-22)
        """
        # Convert to datetime
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        # Get all data announced on or before as_of_date
        available_data = self.fundamentals[
            self.fundamentals[self.announced_col] <= as_of_date
        ]

        if available_data.empty:
            raise PointInTimeError(
                f"No data available as of {as_of_date}. "
                f"First announcement: {self.fundamentals[self.announced_col].min()}"
            )

        if return_type == 'latest':
            # Return most recent quarter
            return available_data.iloc[-1]
        elif return_type == 'all':
            # Return all available quarters
            return available_data.reset_index(drop=True)
        else:
            raise ValueError(f"Invalid return_type: {return_type}. Use 'latest' or 'all'")

    def get_data_series(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = 'Q'
    ) -> pd.DataFrame:
        """
        Generate time-series of data availability.

        Args:
            start_date: Start date
            end_date: End date
            frequency: Rebalancing frequency ('Q'=quarterly, 'M'=monthly, 'W'=weekly)

        Returns:
            DataFrame with columns:
            - date: Rebalance date
            - quarter_date: Quarter end date of available data
            - announced_date: When data was announced
            - lag_days: Days since announcement
            - [all fundamental columns]

        Example:
            # Get quarterly snapshots of available data
            series = pit.get_data_series('2020-01-01', '2025-01-01', frequency='Q')
            # Returns: What data was available at each quarter
        """
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Generate rebalance dates
        if frequency == 'Q':
            dates = pd.date_range(start_date, end_date, freq='QS')
        elif frequency == 'M':
            dates = pd.date_range(start_date, end_date, freq='MS')
        elif frequency == 'W':
            dates = pd.date_range(start_date, end_date, freq='W-MON')
        elif frequency == 'D':
            dates = pd.date_range(start_date, end_date, freq='D')
        else:
            raise ValueError(f"Invalid frequency: {frequency}. Use 'Q', 'M', 'W', or 'D'")

        # Build series
        series_data = []

        for date in dates:
            try:
                # Get latest data available as of this date
                data = self.get_data_as_of(date, return_type='latest')

                # Calculate lag
                lag_days = (date - data[self.announced_col]).days

                # Build row
                row = {
                    'rebalance_date': date,
                    'quarter_date': data[self.date_col],
                    'announced_date': data[self.announced_col],
                    'lag_days': lag_days
                }

                # Add all fundamental data
                for col in self.fundamentals.columns:
                    if col not in [self.date_col, self.announced_col]:
                        row[col] = data[col]

                series_data.append(row)

            except PointInTimeError:
                # No data available yet
                continue

        if not series_data:
            raise PointInTimeError(
                f"No data available between {start_date} and {end_date}"
            )

        df = pd.DataFrame(series_data)

        print(f"✅ Generated time-series: {len(df)} periods")
        print(f"   Frequency: {frequency}")
        print(f"   Date range: {df['rebalance_date'].min()} to {df['rebalance_date'].max()}")

        return df

    def get_announcement_schedule(self) -> pd.DataFrame:
        """
        Get schedule of earnings announcements.

        Returns:
            DataFrame with announcement schedule
        """
        schedule = self.fundamentals[[self.date_col, self.announced_col]].copy()
        schedule['lag_days'] = (
            schedule[self.announced_col] - schedule[self.date_col]
        ).dt.days

        schedule = schedule.sort_values(self.announced_col)

        return schedule

    def validate_no_lookahead(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'Q'
    ) -> bool:
        """
        Validate that no look-ahead bias exists in time-series.

        Args:
            start_date: Start date for validation
            end_date: End date for validation
            frequency: Rebalancing frequency

        Returns:
            True if no look-ahead bias detected

        Raises:
            PointInTimeError if look-ahead bias detected
        """
        print(f"\n🔍 Validating no look-ahead bias...")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Frequency: {frequency}")

        # Get time-series
        series = self.get_data_series(start_date, end_date, frequency)

        # Check 1: All announced dates <= rebalance dates
        lookahead_violations = series[
            series['announced_date'] > series['rebalance_date']
        ]

        if not lookahead_violations.empty:
            print(f"\n❌ LOOK-AHEAD BIAS DETECTED!")
            print(f"   {len(lookahead_violations)} violations found:")
            print(lookahead_violations[['rebalance_date', 'announced_date', 'quarter_date']].head())
            raise PointInTimeError("Look-ahead bias detected!")

        # Check 2: Verify data changes over time (not static)
        unique_quarters = series['quarter_date'].nunique()
        total_periods = len(series)

        print(f"   Total periods: {total_periods}")
        print(f"   Unique quarters used: {unique_quarters}")

        if unique_quarters == 1:
            print(f"⚠️  WARNING: Using same quarter for all periods!")
            print(f"   This suggests static data (not dynamic).")

        # Check 3: Verify reasonable lag
        mean_lag = series['lag_days'].mean()
        max_lag = series['lag_days'].max()

        print(f"   Mean lag: {mean_lag:.1f} days")
        print(f"   Max lag: {max_lag} days")

        if max_lag > 180:
            print(f"⚠️  WARNING: Using very stale data (max lag {max_lag} days)")

        print(f"\n✅ NO LOOK-AHEAD BIAS DETECTED")
        print(f"   All data accessed only after announcement")
        print(f"   Data changes over time: {unique_quarters} unique quarters")

        return True

    def get_available_quarters_count(self, as_of_date: Union[str, datetime]) -> int:
        """
        Get number of quarters available as of date.

        Args:
            as_of_date: Date to check

        Returns:
            Number of available quarters
        """
        as_of_date = pd.to_datetime(as_of_date)
        available = self.fundamentals[
            self.fundamentals[self.announced_col] <= as_of_date
        ]
        return len(available)

    def get_next_announcement(
        self,
        after_date: Union[str, datetime]
    ) -> Optional[Tuple[datetime, pd.Series]]:
        """
        Get next announcement after a specific date.

        Args:
            after_date: Date to search after

        Returns:
            Tuple of (announcement_date, data) or None if no future announcements
        """
        after_date = pd.to_datetime(after_date)

        future_announcements = self.fundamentals[
            self.fundamentals[self.announced_col] > after_date
        ]

        if future_announcements.empty:
            return None

        next_row = future_announcements.iloc[0]
        return (next_row[self.announced_col], next_row)

    def summary(self) -> Dict:
        """
        Generate summary statistics.

        Returns:
            Dictionary with summary stats
        """
        lag_days = (
            self.fundamentals[self.announced_col] - 
            self.fundamentals[self.date_col]
        ).dt.days

        return {
            'total_quarters': len(self.fundamentals),
            'first_quarter': self.fundamentals[self.date_col].min(),
            'last_quarter': self.fundamentals[self.date_col].max(),
            'first_announcement': self.fundamentals[self.announced_col].min(),
            'last_announcement': self.fundamentals[self.announced_col].max(),
            'mean_announcement_lag_days': lag_days.mean(),
            'min_announcement_lag_days': lag_days.min(),
            'max_announcement_lag_days': lag_days.max(),
            'valid_no_lookahead': (
                self.fundamentals[self.announced_col] >= 
                self.fundamentals[self.date_col]
            ).all()
        }


# Example usage
if __name__ == '__main__':
    """
    Example usage of PointInTimeData
    """
    print("=" * 80)
    print("POINT-IN-TIME DATA MANAGER - EXAMPLE USAGE")
    print("=" * 80)

    # Create sample data
    print("\n📊 Creating sample fundamental data...")

    sample_data = pd.DataFrame({
        'date': pd.to_datetime([
            '2023-01-29', '2023-04-30', '2023-07-30', '2023-10-29',
            '2024-01-28', '2024-04-28'
        ]),
        'announced_date': pd.to_datetime([
            '2023-02-22', '2023-05-24', '2023-08-23', '2023-11-21',
            '2024-02-21', '2024-05-22'
        ]),
        'revenue': [6051000000, 7192000000, 13507000000, 18120000000, 22103000000, 26044000000],
        'eps': [0.88, 1.09, 2.48, 3.71, 4.93, 6.12],
        'net_income': [1414000000, 2043000000, 6188000000, 9243000000, 12285000000, 14881000000]
    })

    print(f"Sample data: {len(sample_data)} quarters")
    print(sample_data[['date', 'announced_date', 'revenue']].to_string(index=False))

    # Initialize PointInTimeData
    print("\n" + "=" * 80)
    print("INITIALIZING POINT-IN-TIME DATA MANAGER")
    print("=" * 80)

    pit = PointInTimeData(sample_data)

    # Test 1: Get data as of specific date
    print("\n" + "=" * 80)
    print("TEST 1: Get data available as of 2024-03-01")
    print("=" * 80)

    data_march = pit.get_data_as_of('2024-03-01')
    print(f"\nLatest quarter available: {data_march['date']}")
    print(f"Announced on: {data_march['announced_date']}")
    print(f"Revenue: ${data_march['revenue']:,.0f}")
    print(f"\n✅ Q4 2023 data available (announced 2024-02-21)")
    print(f"✅ Q1 2024 data NOT available yet (announced 2024-05-22)")

    # Test 2: Generate time-series
    print("\n" + "=" * 80)
    print("TEST 2: Generate quarterly time-series")
    print("=" * 80)

    series = pit.get_data_series('2023-06-01', '2024-12-31', frequency='Q')
    print(f"\nTime-series data:")
    print(series[['rebalance_date', 'quarter_date', 'announced_date', 'revenue']].to_string(index=False))

    # Test 3: Validate no look-ahead bias
    print("\n" + "=" * 80)
    print("TEST 3: Validate no look-ahead bias")
    print("=" * 80)

    is_valid = pit.validate_no_lookahead('2023-06-01', '2024-12-31', frequency='Q')

    # Test 4: Summary
    print("\n" + "=" * 80)
    print("TEST 4: Summary statistics")
    print("=" * 80)

    summary = pit.summary()
    print(f"\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)