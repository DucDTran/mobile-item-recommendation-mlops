import pandas as pd
import numpy as np
from datetime import timedelta
import os
from loguru import logger

# CONFIGURATION
DATA_PATH = "../data/raw"
FEATURE_PATH = "../data/features"
os.makedirs(FEATURE_PATH, exist_ok=True)

RAW_USERS = f"{DATA_PATH}/users.parquet"

def process_and_save_features_for_feast(df, target_date_str, window_days=[1, 3]):

    target_ts = pd.Timestamp(target_date_str)
    target_date = target_ts.date()
    event_timestamp = target_ts
    
    logger.info(f"--- Processing Features for Timestamp: {event_timestamp} ---")
    
    # Define History Limit (Strictly features BEFORE the target time)
    history_df = df[df['date'] < target_date]
    
    def get_window_stats(subset_df, prefix):
        item_stats = subset_df.groupby(['item_id', 'behavior_type']).size().unstack(fill_value=0)
        for i in [1, 2, 3, 4]: 
            if i not in item_stats.columns: item_stats[i] = 0
        # Reorder columns to ensure [1, 2, 3, 4] order before renaming
        item_stats = item_stats[[1, 2, 3, 4]]
        item_stats.columns = [f'{prefix}_item_clk', f'{prefix}_item_fav', f'{prefix}_item_cart', f'{prefix}_item_buy']
        # Item Conversion Rate
        item_stats[f'{prefix}_item_cr'] = item_stats[f'{prefix}_item_buy'] / (item_stats[f'{prefix}_item_clk'] + 1)
        # Unique Users per Item
        item_uniq = subset_df.groupby('item_id')['user_id'].nunique().to_frame(f'{prefix}_item_uniq_users')
        item_stats = item_stats.join(item_uniq)
        
        # User Features (Activity)
        user_stats = subset_df.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
        for i in [1, 2, 3, 4]: 
            if i not in user_stats.columns: user_stats[i] = 0
        # Reorder columns to ensure [1, 2, 3, 4] order before renaming
        user_stats = user_stats[[1, 2, 3, 4]]
        user_stats.columns = [f'{prefix}_user_clk', f'{prefix}_user_fav', f'{prefix}_user_cart', f'{prefix}_user_buy']
        
        # User Conversion Rate
        user_stats[f'{prefix}_user_cr'] = user_stats[f'{prefix}_user_buy'] / (user_stats[f'{prefix}_user_clk'] + 1)

        # User-Item Interaction (Specific Interest)
        ui_stats = subset_df.groupby(['user_id', 'item_id', 'behavior_type']).size().unstack(fill_value=0)
        for i in [1, 2, 3, 4]: 
            if i not in ui_stats.columns: ui_stats[i] = 0
        # Reorder columns to ensure [1, 2, 3, 4] order before renaming
        ui_stats = ui_stats[[1, 2, 3, 4]]
        ui_stats.columns = [f'{prefix}_ui_clk', f'{prefix}_ui_fav', f'{prefix}_ui_cart', f'{prefix}_ui_buy']

        # User-Category Interaction (Broader Interest)
        uc_stats = subset_df.groupby(['user_id', 'item_category', 'behavior_type']).size().unstack(fill_value=0)
        for i in [1, 2, 3, 4]: 
            if i not in uc_stats.columns: uc_stats[i] = 0
        # Reorder columns to ensure [1, 2, 3, 4] order before renaming
        uc_stats = uc_stats[[1, 2, 3, 4]]
        uc_stats.columns = [f'{prefix}_uc_clk', f'{prefix}_uc_fav', f'{prefix}_uc_cart', f'{prefix}_uc_buy']

        return item_stats, user_stats, ui_stats, uc_stats

    for days in window_days:
        start_date = target_date - timedelta(days=days)
        window_df = history_df[history_df['date'] >= start_date]
        prefix = f"last_{days}d"
        
        logger.info(f"Calculating {prefix} stats...")
        i_stats, u_stats, ui_stats, uc_stats = get_window_stats(window_df, prefix)
        
        # Save Item Features
        df_i = i_stats.reset_index()
        df_i['event_timestamp'] = event_timestamp
        df_i.to_parquet(f"{FEATURE_PATH}/item_stats_{prefix}_{target_date}.parquet", index=False)

        # Save User Features
        df_u = u_stats.reset_index()
        df_u['event_timestamp'] = event_timestamp
        df_u.to_parquet(f"{FEATURE_PATH}/user_stats_{prefix}_{target_date}.parquet", index=False)

        # Save User-Item Features
        df_ui = ui_stats.reset_index()
        # Create composite key for Feast (doesn't support multiple join keys)
        df_ui['user_item_id'] = df_ui['user_id'].astype(str) + '_' + df_ui['item_id'].astype(str)
        df_ui = df_ui.drop(columns=['user_id', 'item_id'])
        df_ui['event_timestamp'] = event_timestamp
        df_ui.to_parquet(f"{FEATURE_PATH}/ui_stats_{prefix}_{target_date}.parquet", index=False)

        # Save User-Category Features
        df_uc = uc_stats.reset_index()
        # Create composite key for Feast (doesn't support multiple join keys)
        df_uc['user_category_id'] = df_uc['user_id'].astype(str) + '_' + df_uc['item_category'].astype(str)
        df_uc = df_uc.drop(columns=['user_id', 'item_category'])
        df_uc['event_timestamp'] = event_timestamp
        df_uc.to_parquet(f"{FEATURE_PATH}/uc_stats_{prefix}_{target_date}.parquet", index=False)

    # Recency & Time Features
    logger.info("Calculating Recency & Time features...")    
    last_interaction = history_df.sort_values('time').groupby(['user_id', 'item_id']).tail(1).copy()

    # Calculate Hour
    last_interaction['last_touch_hour'] = last_interaction['time'].dt.hour
    
    # Calculate Days Since
    last_interaction['date'] = pd.to_datetime(last_interaction['date'])
    last_interaction['days_since'] = (target_ts - last_interaction['date']).dt.days
    
    # Create composite key for Feast (doesn't support multiple join keys)
    recency_df = last_interaction[['user_id', 'item_id', 'last_touch_hour', 'days_since']].copy()
    recency_df['user_item_id'] = recency_df['user_id'].astype(str) + '_' + recency_df['item_id'].astype(str)
    recency_df = recency_df.drop(columns=['user_id', 'item_id'])
    
    # Add timestamp for Feast
    recency_df['event_timestamp'] = event_timestamp
    
    # Save Recency Features
    recency_df.to_parquet(f"{FEATURE_PATH}/recency_stats_{target_date}.parquet", index=False)
    
    logger.info(f"Success. All Feature Tables saved to {FEATURE_PATH}")

def main():
    if not os.path.exists(RAW_USERS):
        logger.error(f"Raw data not found at {RAW_USERS}")
        return

    logger.info("Loading Raw Parquet Data...")
    df_user = pd.read_parquet(RAW_USERS)
    
    # Ensure types for calculation
    df_user['time'] = pd.to_datetime(df_user['time'])
    df_user['date'] = df_user['time'].dt.date

    # Generate for Test Day (Dec 18) - Simulating Production
    process_and_save_features_for_feast(df_user, "2014-12-18")
    
    # Generate for Validation Day (Dec 12) - For initial evaluation and hyperparameter tuning
    process_and_save_features_for_feast(df_user, "2014-12-12")

    # Generate for Training Day (Dec 6) - For training and backfill
    process_and_save_features_for_feast(df_user, "2014-12-06")

if __name__ == "__main__":
    main()