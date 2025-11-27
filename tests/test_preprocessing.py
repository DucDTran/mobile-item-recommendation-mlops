import pytest
import pandas as pd
import os
from src.preprocessing import process_and_save_features_for_feast, DATA_PATH

@pytest.fixture
def mock_data_paths(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'user_id': [1, 1, 2],
        'item_id': [100, 101, 100],
        'item_category': [10, 10, 10],
        'behavior_type': [1, 4, 1], # 1=Click, 4=Buy
        'time': [
            '2014-12-17 10:00:00', # Day before target
            '2014-12-17 11:00:00', 
            '2014-12-16 10:00:00'  # 2 days before
        ]
    })
    
    raw_file = tmp_path / "users.parquet"
    df.to_parquet(raw_file)
    
    monkeypatch.setattr("src.preprocessing.RAW_USERS", str(raw_file))
    
    out_dir = tmp_path / "features"
    out_dir.mkdir()
    monkeypatch.setattr("src.preprocessing.FEATURE_PATH", str(out_dir))
    
    return out_dir

def test_feature_generation_logic(mock_data_paths):
    df = pd.read_parquet(os.path.join(os.path.dirname(mock_data_paths), "users.parquet"))
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date

    process_and_save_features_for_feast(df, "2014-12-18", window_days=[1])

    expected_file = mock_data_paths / "user_stats_last_1d_2014-12-18.parquet"
    assert expected_file.exists()

    result_df = pd.read_parquet(expected_file)
    
    user1 = result_df[result_df['user_id'] == 1].iloc[0]
    assert user1['last_1d_user_clk'] == 1
    assert user1['last_1d_user_buy'] == 1
    
    if 2 in result_df['user_id'].values:
        user2 = result_df[result_df['user_id'] == 2].iloc[0]
        assert user2['last_1d_user_clk'] == 0