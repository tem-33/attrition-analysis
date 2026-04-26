import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
        "overtime": ["Yes", "No", "Yes", "No", "No", "No"],
        "monthly_income": [3000.0, 6000.0, 4000.0, 8000.0, 5000.0, 7000.0],
        "job_satisfaction": [1, 4, 2, 4, 3, 4],
        "attrition": ["Yes", "No", "Yes", "No", "No", "No"],
    })


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    assert attrition_rate(df) == 50.0


def test_attrition_rate_zero():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["No", "No"],
    })
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["Yes", "Yes"],
    })
    assert attrition_rate(df) == 100.0


def test_attrition_rate_rounds_to_two_decimals():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3],
        "attrition": ["Yes", "No", "No"],
    })
    assert attrition_rate(df) == 33.33


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_correct_rates(sample_df):
    result = attrition_by_department(sample_df)

    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0

    it = result[result["department"] == "IT"].iloc[0]
    assert it["employees"] == 2
    assert it["leavers"] == 0
    assert it["attrition_rate"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_correct_rates(sample_df):
    result = attrition_by_overtime(sample_df)

    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    assert yes_row["employees"] == 2
    assert yes_row["leavers"] == 2
    assert yes_row["attrition_rate"] == 100.0

    no_row = result[result["overtime"] == "No"].iloc[0]
    assert no_row["employees"] == 4
    assert no_row["leavers"] == 0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_correct_values(sample_df):
    result = average_income_by_attrition(sample_df)

    yes_avg = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_avg = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]

    assert yes_avg == 3500.0   # (3000 + 4000) / 2
    assert no_avg == 6500.0    # (6000 + 8000 + 5000 + 7000) / 4


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    scores = result["job_satisfaction"].tolist()
    assert scores == sorted(scores)


def test_satisfaction_summary_rates_divide_by_group_headcount():
    # Score 1 has 2 employees, 1 leaver -> rate should be 50%, not 100%
    # (100% would be the result of the old bug: 1 leaver / 1 total leaver)
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "job_satisfaction": [1, 1, 4, 4],
        "attrition": ["Yes", "No", "No", "No"],
    })
    result = satisfaction_summary(df)

    score_1 = result[result["job_satisfaction"] == 1].iloc[0]
    assert score_1["total_employees"] == 2
    assert score_1["leavers"] == 1
    assert score_1["attrition_rate"] == 50.0


def test_satisfaction_summary_zero_attrition_group(sample_df):
    result = satisfaction_summary(sample_df)
    score_4 = result[result["job_satisfaction"] == 4].iloc[0]
    assert score_4["leavers"] == 0
    assert score_4["attrition_rate"] == 0.0
