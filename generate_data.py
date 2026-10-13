import numpy as np
import pandas as pd

np.random.seed(42)


def generate_niper_data(num_samples=1500):
  campuses = [
      'NIPER SAS Nagar (Mohali)',
      'NIPER Ahmedabad',
      'NIPER Hyderabad',
      'NIPER Guwahati',
      'NIPER Kolkata',
      'NIPER Raebareli',
      'NIPER Hajipur',
  ]
  grant_categories = [
      'Campus Infrastructure & Civil',
      'R&D Equipment & Instruments',
      'Consumables & Operational Grants',
      'Clinical Trial & Testing Facilities',
      'Faculty Research & Fellowships',
  ]
  quarters = ['Q1', 'Q2', 'Q3', 'Q4']
  fiscal_years = [f'{year}-{year+1}' for year in range(2016, 2026)]

  campus_col = np.random.choice(campuses, size=num_samples)
  grant_col = np.random.choice(grant_categories, size=num_samples)
  quarter_col = np.random.choice(
      quarters, size=num_samples, p=[0.20, 0.30, 0.35, 0.15]
  )
  fy_col = np.random.choice(fiscal_years, size=num_samples)

  budget_allocated_cr = np.round(np.random.uniform(2.0, 50.0, num_samples), 2)
  release_delay_days = np.random.randint(0, 121, num_samples)
  procurement_completion_pct = np.round(
      np.random.uniform(15.0, 100.0, num_samples), 2
  )
  prior_year_utilization_pct = np.round(
      np.random.uniform(50.0, 98.0, num_samples), 2
  )
  active_projects_count = np.random.randint(3, 46, num_samples)

  # Domain Logic for Target Variable
  campus_bias = {
      'NIPER SAS Nagar (Mohali)': 5.0,
      'NIPER Hyderabad': 4.0,
      'NIPER Ahmedabad': 3.5,
      'NIPER Guwahati': 1.0,
      'NIPER Kolkata': 0.0,
      'NIPER Raebareli': -2.5,
      'NIPER Hajipur': -4.0,
  }
  quarter_map = {'Q1': 4.0, 'Q2': 2.0, 'Q3': -3.0, 'Q4': -10.0}

  campus_effect = np.array([campus_bias[c] for c in campus_col])
  quarter_effect = np.array([quarter_map[q] for q in quarter_col])
  delay_effect = -1.5 * (release_delay_days / 10.0)
  procurement_effect = 0.25 * (procurement_completion_pct - 50.0)
  momentum_effect = 0.20 * (prior_year_utilization_pct - 75.0)
  noise = np.random.normal(0, 4.5, num_samples)

  raw_utilization = (
      80.0
      + campus_effect
      + quarter_effect
      + delay_effect
      + procurement_effect
      + momentum_effect
      + noise
  )
  actual_utilization_pct = np.round(np.clip(raw_utilization, 25.0, 99.5), 2)
  actual_expenditure_cr = np.round(
      (actual_utilization_pct / 100.0) * budget_allocated_cr, 2
  )

  risk_level = np.select(
      [
          actual_utilization_pct < 70.0,
          (actual_utilization_pct >= 70.0) & (actual_utilization_pct <= 85.0),
          actual_utilization_pct > 85.0,
      ],
      ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK'],
      default='MEDIUM_RISK',
  )

  df = pd.DataFrame({
      'fiscal_year': fy_col,
      'campus_name': campus_col,
      'grant_category': grant_col,
      'quarter_released': quarter_col,
      'budget_allocated_cr': budget_allocated_cr,
      'release_delay_days': release_delay_days,
      'procurement_completion_pct': procurement_completion_pct,
      'prior_year_utilization_pct': prior_year_utilization_pct,
      'active_projects_count': active_projects_count,
      'actual_expenditure_cr': actual_expenditure_cr,
      'actual_utilization_pct': actual_utilization_pct,
      'risk_level': risk_level,
  })

  df.to_csv('niper_budget_data.csv', index=False)
  print('Successfully generated niper_budget_data.csv!')


if __name__ == '__main__':
  generate_niper_data()