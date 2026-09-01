# Dataset: Orders

## Description

Synthetic e-commerce orders dataset used for model training, registration, and inference in the MLflow pipeline.

- Source: CSV at `data/orders.csv`
- Use case: classify whether an order is `delivered` (`1`) vs not delivered (`0`)
- Shape: 10,000 rows x 12 columns

## Columns

- `order_id`: Order identifier string (example `ORD-0000001`); unique key, not used as a feature.
- `customer_id`: Customer identifier string (example `CUST-001502`); high cardinality, not used as a feature.
- `order_date`: Timestamp when the order was placed (`YYYY-MM-DD HH:MM:SS.ssssss`); used to derive time features.
- `status`: Order lifecycle status. Values: `cancelled`, `delivered`, `in_transit`, `pending`, `processing`, `returned`, `shipped`. Target source column.
- `num_items`: Number of items in the order. Numeric range in dataset: 1 to 5.
- `subtotal`: Pre-tax/sub-shipping monetary subtotal. Numeric range: 11.01 to 7832.25.
- `tax`: Tax amount. Numeric range: 0.88 to 626.58.
- `shipping`: Shipping fee. Numeric range: 0.00 to 9.99.
- `total`: Final charged amount. Numeric range: 21.88 to 8458.83.
- `payment_method`: Categorical payment type. Values: `apple_pay`, `credit_card`, `debit_card`, `paypal`.
- `shipping_address`: Free-text shipping address string; not used as a feature.
- `delivery_date`: Delivery timestamp for fulfilled orders; 1,826 missing values (typically non-delivered statuses), not used as a feature.
