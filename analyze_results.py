import pandas as pd

if __name__ == "__main__":
    low_frequency_results = pd.read_csv("rmse_ny_fc_[6000]_antennas_[8]_bw_[6]_subcarriers_[24].csv", index_col=0)
    high_frequency_results = pd.read_csv("rmse_ny_fc_[24000]_antennas_[32]_bw_[12]_subcarriers_[48].csv", index_col=0)
    combined = low_frequency_results.iloc[:-1].join(high_frequency_results.iloc[:-1], lsuffix='_low', rsuffix='_high')
    confusion_matrix = pd.crosstab(combined['Error > 1m_low'], combined['Error > 1m_high'])
    print(confusion_matrix)
