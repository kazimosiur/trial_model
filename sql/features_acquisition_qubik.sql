---
# UDF to obtain min / max order dates. Need this because customer table has QC split into Dark + Local. UDF ignores NULL values (problem with built-in BQ functions is they will return NULL if it exists)
CREATE TEMP FUNCTION LEAST_ARRAY(arr ANY TYPE) AS ((
    SELECT MIN(a) FROM UNNEST(arr) a WHERE a IS NOT NULL));
CREATE TEMP FUNCTION GREATEST_ARRAY(arr ANY TYPE) AS ((
    SELECT MAX(a) FROM UNNEST(arr) a WHERE a IS NOT NULL));
---

# Aggregation of DarkS / LocalS into QC
WITH
qc_agg AS (
  SELECT
    global_entity_id
    ,analytical_customer_id
    ,LEAST(orders.darkstores.first_order.order_date, orders.localstores.first_order.order_date)                      AS first_order_date_vert_qc
    ,COALESCE(orders.darkstores.lifetime.order_count,0) + COALESCE(orders.localstores.lifetime.order_count,0)        AS qc_lifetime_order_count
    ,COALESCE(orders.darkstores.lifetime.late_orders,0) + COALESCE(orders.localstores.lifetime.late_orders,0)        AS qc_lifetime_order_count_late
    ,COALESCE(SAFE_DIVIDE(
        COALESCE(orders.darkstores.lifetime.gmv_sum, 0) + COALESCE(orders.localstores.lifetime.gmv_sum, 0),
        COALESCE(orders.darkstores.lifetime.order_count, 0) + COALESCE(orders.localstores.lifetime.order_count, 0)
    ), 0)                                                                                                            AS qc_lifetime_gmv_avg

  FROM `fulfillment-dwh-production.cl_dmart.customers`
  WHERE global_entity_id = '{GLOBAL_ENTITY_ID}'

),

order_data AS (
  SELECT
     cust.global_entity_id
    ,cust.analytical_customer_id
    --
    ,cust.orders.restaurants.first_order.order_date                AS first_order_date_vert_rs
    ,agg.first_order_date_vert_qc
    --
    ,DATE_DIFF(CURRENT_DATE() - 1, cust.orders.restaurants.first_order.order_date, DAY) + 1 AS days_since_first_order_vert_rs
    ,DATE_DIFF(CURRENT_DATE() - 1, agg.first_order_date_vert_qc, DAY) + 1                   AS days_since_first_order_vert_qc
    ,DATE_DIFF(CURRENT_DATE() - 1, cust.orders.restaurants.last_order.order_date, DAY) + 1  AS days_since_last_order_vert_rs
    --
    ,EXTRACT(DAYOFWEEK FROM cust.orders.restaurants.first_order.order_date)           AS first_order_dayofweek_vert_rs
    ,cust.orders.restaurants.first_order.gmv                                          AS first_order_amt_gmv_eur_vert_rs
    --
    ,cust.orders.restaurants.first_order.delivery_fee                                 AS first_order_delivery_fee_vert_rs
    ,COALESCE(SAFE_DIVIDE(cust.orders.restaurants.first_order.delivery_fee,
                          cust.orders.restaurants.first_order.gmv)          , 0)      AS first_order_delivery_fee_pct_vert_rs
    ,cust.orders.restaurants.first_order.delivery_distnace                            AS first_order_delivery_dist_m_vert_rs
    ,cust.orders.restaurants.first_order.delivery_time_promised                       AS first_order_delivery_time_promised_vert_rs
    ,cust.orders.restaurants.first_order.delivery_delay                               AS first_order_delivery_delay_vert_rs
    --
    ,cust.orders.restaurants.first_order.discount                                     AS first_order_discount_vert_rs
    ,COALESCE(SAFE_DIVIDE(cust.orders.restaurants.first_order.discount,
                          cust.orders.restaurants.first_order.gmv)          , 0)      AS first_order_discount_pct_vert_rs   
    ,cust.orders.restaurants.first_order.voucher                                      AS first_order_voucher_vert_rs
    ,COALESCE(SAFE_DIVIDE(cust.orders.restaurants.first_order.voucher,
                          cust.orders.restaurants.first_order.gmv)          , 0)      AS first_order_voucher_pct_vert_rs
    --- 
    ,CASE
       WHEN
            orders.darkstores.first_order.order_date < orders.localstores.first_order.order_date
       THEN orders.darkstores.first_order.voucher
       WHEN
            orders.darkstores.first_order.order_date >= orders.localstores.first_order.order_date
       THEN orders.localstores.first_order.voucher
    END                                                                               AS first_order_voucher_vert_qc
    ---
    ,cust.location.last_order_location.last_order_latitude                            AS cust_lat                                                   
    ,cust.location.last_order_location.last_order_longitude                           AS cust_long                                                   
    ---
    ,SAFE_DIVIDE(cust.orders.restaurants.lifetime.weekend_order_count, cust.orders.restaurants.lifetime.order_count)                AS order_perc_weekend_vert_rs
    ,SAFE_DIVIDE(cust.orders.restaurants.lifetime.morning_order_count, cust.orders.restaurants.lifetime.order_count)                AS order_perc_morning_vert_rs
    ,SAFE_DIVIDE(cust.orders.restaurants.lifetime.afternoon_order_count, cust.orders.restaurants.lifetime.order_count)              AS order_perc_afternoon_vert_rs
    ,SAFE_DIVIDE(cust.orders.restaurants.lifetime.night_order_count, cust.orders.restaurants.lifetime.order_count)                  AS order_perc_night_vert_rs
    ---
    ,cust.orders.restaurants.lifetime.order_count         AS order_count_lifetime_vert_rs
    ,agg.qc_lifetime_order_count                          AS order_count_lifetime_vert_qc
    ,cust.orders.restaurants.last4w.order_count           AS order_count_l04w_vert_rs
    ,cust.orders.restaurants.last16w.order_count          AS order_count_l16w_vert_rs    
    ---
    ,cust.orders.restaurants.lifetime.late_orders         AS order_count_late_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.late_orders           AS order_count_late_l04w_vert_rs
    ,cust.orders.restaurants.last16w.late_orders          AS order_count_late_l16w_vert_rs 
    ---
    ,ROUND(SAFE_DIVIDE(cust.orders.restaurants.last16w.late_orders,  cust.orders.restaurants.last16w.order_count), 2)  AS order_pct_late_l16w_vert_rs
    ,ROUND(SAFE_DIVIDE(cust.orders.restaurants.lifetime.late_orders, cust.orders.restaurants.lifetime.order_count), 2) AS order_pct_late_lifetime_vert_rs
    ---
    ,cust.orders.restaurants.lifetime.gmv_sum         AS gmv_sum_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.gmv_sum           AS gmv_sum_l04w_vert_rs
    ,cust.orders.restaurants.last16w.gmv_sum          AS gmv_sum_l16w_vert_rs     
    ---
    ,cust.orders.restaurants.lifetime.gmv_avg         AS gmv_avg_lifetime_vert_rs
    ,qc_lifetime_gmv_avg                              AS gmv_avg_lifetime_vert_qc
    ,cust.orders.restaurants.last4w.gmv_avg           AS gmv_avg_l04w_vert_rs
    ,cust.orders.restaurants.last16w.gmv_avg          AS gmv_avg_l16w_vert_rs     
    ---
    ,cust.orders.restaurants.lifetime.delivery_fee_sum         AS delivery_fee_sum_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.delivery_fee_sum           AS delivery_fee_sum_l04w_vert_rs
    ,cust.orders.restaurants.last16w.delivery_fee_sum          AS delivery_fee_sum_l16w_vert_rs     
    ---
    ,cust.orders.restaurants.lifetime.delivery_fee_avg         AS delivery_fee_avg_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.delivery_fee_avg           AS delivery_fee_avg_l04w_vert_rs
    ,cust.orders.restaurants.last16w.delivery_fee_avg          AS delivery_fee_avg_l16w_vert_rs     
    ---  
    ,cust.orders.restaurants.lifetime.discount_sum         AS discount_sum_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.discount_sum           AS discount_sum_l04w_vert_rs
    ,cust.orders.restaurants.last16w.discount_sum          AS discount_sum_l16w_vert_rs     
    ---
    ,cust.orders.restaurants.lifetime.discount_avg         AS discount_avg_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.discount_avg           AS discount_avg_l04w_vert_rs
    ,cust.orders.restaurants.last16w.discount_avg          AS discount_avg_l16w_vert_rs     
    ---  
    ,cust.orders.restaurants.lifetime.voucher_sum         AS voucher_sum_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.voucher_sum           AS voucher_sum_l04w_vert_rs
    ,cust.orders.restaurants.last16w.voucher_sum          AS voucher_sum_l16w_vert_rs     
    ---
    ,cust.orders.restaurants.lifetime.voucher_avg         AS voucher_avg_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.voucher_avg           AS voucher_avg_l04w_vert_rs
    ,cust.orders.restaurants.last16w.voucher_avg          AS voucher_avg_l16w_vert_rs     
    ---  
    ,cust.orders.restaurants.lifetime.joker_cust_discount_sum         AS joker_cust_discount_sum_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.joker_cust_discount_sum           AS joker_cust_discount_sum_l04w_vert_rs
    ,cust.orders.restaurants.last16w.joker_cust_discount_sum          AS joker_cust_discount_sum_l16w_vert_rs     
    ---
    ,cust.orders.restaurants.lifetime.joker_vendor_fee_sum         AS joker_vendor_fee_sum_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.joker_vendor_fee_sum           AS joker_vendor_fee_sum_l04w_vert_rs
    ,cust.orders.restaurants.last16w.joker_vendor_fee_sum          AS joker_vendor_fee_sum_l16w_vert_rs     
    ---      
    ,cust.orders.restaurants.lifetime.deliv_dist_m_avg         AS deliv_dist_m_avg_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.deliv_dist_m_avg           AS deliv_dist_m_avg_l04w_vert_rs
    ,cust.orders.restaurants.last16w.deliv_dist_m_avg          AS deliv_dist_m_avg_l16w_vert_rs     
    ---
    ,cust.orders.restaurants.lifetime.deliv_time_promised_avg         AS deliv_time_promised_avg_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.deliv_time_promised_avg           AS deliv_time_promised_avg_l04w_vert_rs
    ,cust.orders.restaurants.last16w.deliv_time_promised_avg          AS deliv_time_promised_avg_l16w_vert_rs     
    ---      
    ,cust.orders.restaurants.lifetime.delivery_delay_avg         AS delivery_delay_avg_lifetime_vert_rs
    ,cust.orders.restaurants.last4w.delivery_delay_avg           AS delivery_delay_avg_l04w_vert_rs
    ,cust.orders.restaurants.last16w.delivery_delay_avg          AS delivery_delay_avg_l16w_vert_rs     
    ---          
    ,cust.location.coverage_metrics.coverage_vendors_qc
    ,cust.location.coverage_metrics.coverage_dmart
    ,cust.location.coverage_metrics.coverage_localstore
    ,cust.location.coverage_metrics.coverage_convenience
    ,cust.location.coverage_metrics.coverage_groceries
    ,cust.location.coverage_metrics.coverage_supermarket
    ---
    ,cust.location.coverage_metrics.coverage_vendors_qc_in_2000m
    ,cust.location.coverage_metrics.coverage_vendors_qc_in_4000m
    ,cust.location.coverage_metrics.coverage_vendors_qc_in_6000m
    ---
    ,cust.location.coverage_metrics.dist_nearest_vendor_qc
    ,cust.location.coverage_metrics.dist_nearest_dmart
    ,cust.location.coverage_metrics.dist_nearest_localstore

  FROM `fulfillment-dwh-production.cl_dmart.customers` as cust
  LEFT JOIN qc_agg    AS agg      ON cust.global_entity_id = agg.global_entity_id AND cust.analytical_customer_id = agg.analytical_customer_id

  WHERE cust.global_entity_id = '{GLOBAL_ENTITY_ID}'
    AND orders.restaurants.first_order.order_date < CURRENT_DATE() - 1 -  14      # allow 2 weeks between first order and possible QC order
    AND orders.restaurants.last_order.order_date > CURRENT_DATE() - 1 - 180       # any order in last 180d
)

,qubik_base AS (
  SELECT
     p.timestamp
    ,p.global_entity_id
    ,map.analytical_customer_id
    --
    ,visit_attributes.unique_vendors_viewed_last_4_weeks
    ,visit_attributes.avg_unique_vendors_available_last_4_weeks
    ,visit_attributes.unique_addresses_used_last_4_weeks
    ,visit_attributes.visits_last_4_weeks
    ,visit_attributes.visits_last_12_weeks
    ,visit_attributes.session_duration_last_4_weeks
    ,visit_attributes.avg_interaction_speed_last_4_weeks
    ,visit_attributes.abandoned_cart_rate_last_4_weeks
    ,visit_attributes.search_fail_rate_last_4_weeks
    ,visit_attributes.voucher_error_rate_last_4_weeks
    --
    ,order_attributes.first_order_timestamp
    ,order_attributes.online_payment_order_rate_last_4_weeks
    ,order_attributes.avg_ordered_vendors_rating_last_4_weeks
    ,order_attributes.last_after_order_survey_response_score
    ,order_attributes.last_after_order_survey_response_avg_rating
    ,gcc.customer_cc_interactions.total_cc_sessions_last_4_weeks

  FROM       `fulfillment-dwh-production.curated_data_shared_cdp.customer_profiles_v1`        AS p
  INNER JOIN `fulfillment-dwh-production.curated_data_shared_mkt.external_id_to_acid_mapping` AS map
    ON p.global_entity_id = map.global_entity_id AND p.customer.customer_mapping_id = map.external_id
  WHERE calculation_date = CURRENT_DATE-3
    AND  p.global_entity_id = '{GLOBAL_ENTITY_ID}'
)

,qubik_agg AS (
   SELECT
      qc.global_entity_id
     ,qc.analytical_customer_id
     --
     ,MIN(qub.first_order_timestamp)                       AS order_first_ts_qubik
     ,MAX(qub.online_payment_order_rate_last_4_weeks)      AS order_rate_online_payment_l04w
     --
     ,MAX(qub.unique_vendors_viewed_last_4_weeks)          AS visit_vendors_viewed_l04w
     ,MAX(qub.avg_unique_vendors_available_last_4_weeks)   AS visit_vendors_available_l04w
     ,MAX(qub.unique_addresses_used_last_4_weeks)          AS visit_addresses_unique_l04w
     ,MAX(qub.visits_last_4_weeks)                         AS visit_count_l04w
     ,MAX(qub.visits_last_12_weeks)                        AS visit_count_l12w
     ,SAFE_DIVIDE(MAX(qub.visits_last_4_weeks),
                  MAX(qub.visits_last_12_weeks))           AS visit_count_l04w_vs_l12w
     ,MAX(qub.session_duration_last_4_weeks)               AS visit_session_dur_sum_l04w
     ,SAFE_DIVIDE(MAX(qub.session_duration_last_4_weeks),
                  MAX(qub.visits_last_4_weeks))            AS visit_session_dur_avg_l04w
     ,MAX(qub.avg_interaction_speed_last_4_weeks)          AS visit_interact_speed_avg_l04w
     ,MAX(qub.abandoned_cart_rate_last_4_weeks)            AS visit_cart_abandon_rate_l04w
     ,MAX(qub.search_fail_rate_last_4_weeks)               AS visit_search_fail_rate_l04w
     ,MAX(qub.voucher_error_rate_last_4_weeks)             AS visit_voucher_error_rate_l04w
     --
     ,MAX(qub.avg_ordered_vendors_rating_last_4_weeks)     AS rating_avg_l04w
     ,MAX(qub.last_after_order_survey_response_score)      AS aos_score_last
     ,MAX(qub.last_after_order_survey_response_avg_rating) AS aos_rating_last
     --
     ,COALESCE(MAX(qub.total_cc_sessions_last_4_weeks), 0) AS ccc_sessions_l04w
   ----
   FROM      order_data   AS qc
   LEFT JOIN qubik_base                                    AS qub ON  qc.global_entity_id = qub.global_entity_id
                                                                  AND qc.analytical_customer_id = qub.analytical_customer_id
   WHERE qub.first_order_timestamp IS NOT NULL
   GROUP BY 1,2
)

,qubik_data AS (
  SELECT
     global_entity_id
    ,analytical_customer_id
    ,order_rate_online_payment_l04w
    --
    ,visit_vendors_viewed_l04w
    ,visit_vendors_available_l04w
    ,visit_addresses_unique_l04w
    ,visit_count_l04w
    ,visit_count_l12w
    ,visit_count_l04w_vs_l12w
    ,visit_session_dur_sum_l04w
    ,visit_session_dur_avg_l04w
    ,visit_interact_speed_avg_l04w
    ,visit_cart_abandon_rate_l04w
    ,visit_search_fail_rate_l04w
    ,visit_voucher_error_rate_l04w
    ,rating_avg_l04w
    ,aos_score_last
    ,aos_rating_last
    ,ccc_sessions_l04w

  FROM qubik_agg
)

------
SELECT
   l.* 
  ,SAFE_DIVIDE(l.order_count_lifetime_vert_qc, l.days_since_first_order_vert_qc) * 28   AS order_freq_4w_vert_qc
  --
  ,IF(COALESCE(l.order_count_l04w_vert_rs, 0)=0, NULL, order_rate_online_payment_l04w)  AS order_rate_online_payment_l04w
  ,q.visit_vendors_available_l04w
  ,COALESCE(q.visit_vendors_viewed_l04w, 0)                                             AS visit_vendors_viewed_l04w
  ,COALESCE(q.visit_addresses_unique_l04w, 0)                                           AS visit_addresses_unique_l04w
  ,COALESCE(q.visit_count_l04w, 0)                                                      AS visit_count_l04w
  ,COALESCE(q.visit_count_l12w, 0)                                                      AS visit_count_l12w
  ,q.visit_count_l04w_vs_l12w
  ,COALESCE(q.visit_session_dur_sum_l04w, 0)                                            AS visit_session_dur_avg_l04w
  ,q.visit_session_dur_sum_l04w
  ,q.visit_interact_speed_avg_l04w
  ,q.visit_cart_abandon_rate_l04w
  ,q.visit_search_fail_rate_l04w
  ,IF(COALESCE(l.order_count_l04w_vert_rs, 0)=0, NULL, q.visit_voucher_error_rate_l04w) AS visit_voucher_error_rate_l04w
  ,IF(COALESCE(l.order_count_l04w_vert_rs, 0)=0, NULL, q.rating_avg_l04w)               AS rating_avg_l04w
  ,IF(COALESCE(l.order_count_l04w_vert_rs, 0)=0, NULL, q.aos_score_last)                AS aos_score_last
  ,IF(COALESCE(l.order_count_l04w_vert_rs, 0)=0, NULL, q.aos_rating_last)               AS aos_rating_last
  ,IF(COALESCE(l.order_count_l04w_vert_rs, 0)=0,    0, q.ccc_sessions_l04w)             AS ccc_sessions_l04w

----

FROM      order_data                 AS l
LEFT JOIN qubik_data                 AS q USING(global_entity_id, analytical_customer_id)

WHERE order_count_lifetime_vert_rs > 0
  AND coverage_vendors_qc>0
  AND (   SAFE_DIVIDE(order_count_lifetime_vert_qc, days_since_first_order_vert_qc) * 28 < 40 -- remove at 99.9th percentile
       OR SAFE_DIVIDE(order_count_lifetime_vert_qc, days_since_first_order_vert_qc) * 28 IS NULL)
  AND (   order_count_l16w_vert_rs IS NULL
       OR order_count_l16w_vert_rs < 100) -- remove at 99.9th percentile
ORDER BY l.analytical_customer_id
