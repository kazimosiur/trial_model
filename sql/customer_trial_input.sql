SELECT
    global_entity_id
    ,analytical_customer_id
    --
    ,orders.restaurants.    first_order.order_date        AS first_order_date_vert_rs
    ,orders.qc.darkstores.  first_order.order_date        AS first_order_date_vert_dm
    ,orders.qc.local_stores.first_order.order_date        AS first_order_date_vert_ls
    ,LEAST(orders.qc.darkstores.  first_order.order_date,
        orders.qc.local_stores.first_order.order_date)    AS first_order_date_vert_qc
    -- first order restaurant
    # ,orders.restaurants.first_order.order_hour
    # ,orders.restaurants.first_order.order_weekday
    ,orders.restaurants.first_order.gmv                   AS first_order_gmv_eur_vert_rs
    #,orders.restaurants.first_order.discount
    #,orders.restaurants.first_order.discount_pct
    #,orders.restaurants.first_order.voucher
    #,orders.restaurants.first_order.voucher_pct
    ,orders.restaurants.first_order.delivery_delay        AS first_order_delivery_delay_vert_rs
    #,orders.restaurants.first_order.delivery_time_promised
    #,orders.restaurants.first_order.delivery_fee
    #,orders.restaurants.first_order.delivery_fee_pct
    #,orders.restaurants.first_order.delivery_dist
    
    -- first order qc, target
    #,orders.qc.first_order.darkstores.  voucher
    #,orders.qc.first_order.local_stores.voucher

    -- lifetime, restaurant
    ,orders.restaurants.lifetime.order_count              AS order_count_all_lifetime_vert_rs
    ,orders.restaurants.lifetime.late_orders              AS order_count_late_lifetime_vert_rs
    ,orders.restaurants.lifetime.gmv_sum                  AS gmv_sum_lifetime_vert_rs
    ,orders.restaurants.lifetime.gmv_avg                  AS gmv_avg_lifetime_vert_rs
    ,orders.restaurants.lifetime.discount_sum             AS discount_sum_lifetime_vert_rs
    ,orders.restaurants.lifetime.voucher_sum              AS voucher_sum_lifetime_vert_rs
    ,orders.restaurants.lifetime.delivery_fee_sum         AS delivery_fee_sum_lifetime_vert_rs
    ,orders.restaurants.lifetime.delivery_delay_avg       AS delivery_time_delay_avg_lifetime_vert_rs
    ,orders.restaurants.lifetime.deliv_time_promised_avg  AS delivery_time_promised_lifetime_vert_rs
    ,orders.restaurants.lifetime.deliv_dist_m_avg	      AS delivery_distance_avg_lifetime_vert_rs
    
    -- last 16 weeks, restaurant
    ,orders.restaurants.last16w.order_count              AS order_count_all_last16w_vert_rs
    ,orders.restaurants.last16w.late_orders              AS order_count_late_last16w_vert_rs
    ,orders.restaurants.last16w.gmv_sum                  AS gmv_sum_last16w_vert_rs
    ,orders.restaurants.last16w.gmv_avg                  AS gmv_avg_last16w_vert_rs
    ,orders.restaurants.last16w.discount_sum             AS discount_sum_last16w_vert_rs
    ,orders.restaurants.last16w.voucher_sum              AS voucher_sum_last16w_vert_rs
    ,orders.restaurants.last16w.delivery_fee_sum         AS delivery_fee_sum_last16w_vert_rs
    ,orders.restaurants.last16w.delivery_delay_avg       AS delivery_time_delay_avg_last16w_vert_rs
    ,orders.restaurants.last16w.deliv_time_promised_avg  AS delivery_time_promised_last16w_vert_rs
    ,orders.restaurants.last16w.deliv_dist_m_avg	     AS delivery_distance_avg_last16w_vert_rs

    -- ratings
    ,ratings.restaurants.first_rating                    AS rating_first_vert_rs
    ,ratings.restaurants.rating_count                    AS rating_count_vert_rs
    ,ratings.restaurants.rating_average                  AS rating_avg_vert_rs
    ,ratings.restaurants.normalised_rating               AS rating_avg_normalised_vert_rs
    ,ratings.restaurants.distinct_vendor_ratings_count   AS rating_vendor_countd_vert_rs

    -- visits
    ,visits.first_platform_device                        AS visits_first_platform_device
    ,visits.last_platform_device                         AS visits_last_platform_device
    #,visits.last_visit_timestamp                         AS visits_last_visit_timestamp
    ,visits.lifetime_visit_count                         AS visits_count_lifetime
    ,visits.last4w.visits_count                          AS visits_count_last4w
    ,visits.last4w.sum_session_duration                  AS visits_session_duration_sum
    ,visits.last4w.average_session_duration              AS visits_session_duration_avg
    ,visits.last4w.abandoned_cart_rate                   AS visits_cart_abandonment_rate
    ,visits.last4w.qc_visit_rate                         AS visits_qc_visit_rate # potential overfitting?

    -- vendors, restaurants
    ,vendors.restaurants.favorite_cuisine                AS vendor_favorite_cuisine_vert_rs
    ,vendors.restaurants.distinct_vendor_orders_last16w  AS vendor_distinct_orders_l16w_vert_rs
    ,vendors.restaurants.distinct_vendor_orders_last4w   AS vendor_distinct_orders_l4w_vert_rs
    ,vendors.restaurants.restaurant_loyalty              AS vendor_loyalty_vert_rs # look into logic to avoid overfitting
FROM `fulfillment-dwh-production.cl_dmart.customers`
WHERE global_entity_id = '{GLOBAL_ENTITY_ID}'
