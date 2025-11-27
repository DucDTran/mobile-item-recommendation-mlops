from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Int64, Float32, Int32

# Entities
user = Entity(name="user", join_keys=["user_id"])
item = Entity(name="item", join_keys=["item_id"])
# Note: Feast doesn't support multiple join keys, so we use composite keys
user_item = Entity(name="user_item", join_keys=["user_item_id"])
user_category = Entity(name="user_category", join_keys=["user_category_id"])

# Path Pattern for Docker
path_prefix = "/app/data/features"

# Items Statistics (1 Day)
item_stats_1d_view = FeatureView(
    name="item_stats_1d",
    entities=[item],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_1d_item_clk", dtype=Float32),
        Field(name="last_1d_item_fav", dtype=Float32),
        Field(name="last_1d_item_cart", dtype=Float32),
        Field(name="last_1d_item_buy", dtype=Float32),
        Field(name="last_1d_item_cr", dtype=Float32),
        Field(name="last_1d_item_uniq_users", dtype=Int64),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/item_stats_last_1d_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)

# User Statistics (1 Day)
user_stats_1d_view = FeatureView(
    name="user_stats_1d",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_1d_user_clk", dtype=Float32),
        Field(name="last_1d_user_fav", dtype=Float32),
        Field(name="last_1d_user_cart", dtype=Float32),
        Field(name="last_1d_user_buy", dtype=Float32),
        Field(name="last_1d_user_cr", dtype=Float32),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/user_stats_last_1d_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)

# User-Item Statistics (1 Day)
ui_stats_1d_view = FeatureView(
    name="ui_stats_1d",
    entities=[user_item],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_1d_ui_clk", dtype=Float32),
        Field(name="last_1d_ui_fav", dtype=Float32),
        Field(name="last_1d_ui_cart", dtype=Float32),
        Field(name="last_1d_ui_buy", dtype=Float32),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/ui_stats_last_1d_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)

# User-Category Statistics (1 Day)
uc_stats_1d_view = FeatureView(
    name="uc_stats_1d",
    entities=[user_category],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_1d_uc_clk", dtype=Float32),
        Field(name="last_1d_uc_fav", dtype=Float32),
        Field(name="last_1d_uc_cart", dtype=Float32),
        Field(name="last_1d_uc_buy", dtype=Float32),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/uc_stats_last_1d_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)

# Recency & Time (User-Item)
recency_stats_view = FeatureView(
    name="recency_stats",
    entities=[user_item],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_touch_hour", dtype=Int32),
        Field(name="days_since", dtype=Int64),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/recency_stats_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)

# Items Statistics (3 Days)
item_stats_3d_view = FeatureView(
    name="item_stats_3d",
    entities=[item],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_3d_item_clk", dtype=Float32),
        Field(name="last_3d_item_fav", dtype=Float32),
        Field(name="last_3d_item_cart", dtype=Float32),
        Field(name="last_3d_item_buy", dtype=Float32),
        Field(name="last_3d_item_cr", dtype=Float32),
        Field(name="last_3d_item_uniq_users", dtype=Int64),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/item_stats_last_3d_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)

# User Statistics (3 Days)
user_stats_3d_view = FeatureView(
    name="user_stats_3d",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_3d_user_clk", dtype=Float32),
        Field(name="last_3d_user_fav", dtype=Float32),
        Field(name="last_3d_user_cart", dtype=Float32),
        Field(name="last_3d_user_buy", dtype=Float32),
        Field(name="last_3d_user_cr", dtype=Float32),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/user_stats_last_3d_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)

# User-Item Statistics (3 Days)
ui_stats_3d_view = FeatureView(
    name="ui_stats_3d",
    entities=[user_item],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_3d_ui_clk", dtype=Float32),
        Field(name="last_3d_ui_fav", dtype=Float32),
        Field(name="last_3d_ui_cart", dtype=Float32),
        Field(name="last_3d_ui_buy", dtype=Float32),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/ui_stats_last_3d_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)

# User-Category Statistics (3 Days)
uc_stats_3d_view = FeatureView(
    name="uc_stats_3d",
    entities=[user_category],
    ttl=timedelta(days=365),
    schema=[
        Field(name="last_3d_uc_clk", dtype=Float32),
        Field(name="last_3d_uc_fav", dtype=Float32),
        Field(name="last_3d_uc_cart", dtype=Float32),
        Field(name="last_3d_uc_buy", dtype=Float32),
    ],
    online=True,
    source=FileSource(
        path=f"{path_prefix}/uc_stats_last_3d_2014-12-18.parquet",
        timestamp_field="event_timestamp"
    )
)
