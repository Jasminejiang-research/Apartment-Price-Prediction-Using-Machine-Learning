# Appartments_train

The dataset consists of apartment records with the following features:

unit_id – Unique (and anonymized) identifier for each apartment.
obj_type – Type of apartment or object (categorical, anonymized).
dim_m2 – Apartment size in square meters.
n_rooms – Number of rooms.
floor_no – The floor on which the apartment is located.
floor_max – Total number of floors in the building.
year_built – The year the building was constructed.
dist_centre – Distance from the apartment to the city center.
n_poi – Number of points of interest nearby.
dist_sch – Distance to the nearest school.
dist_clinic – Distance to the nearest clinic.
dist_post – Distance to the nearest post office.
dist_kind – Distance to the nearest kindergarten.
dist_rest – Distance to the nearest restaurant.
dist_uni – Distance to the nearest college or university.
dist_pharma – Distance to the nearest pharmacy.
own_type – Ownership type (categorical, anonymized).
build_mat – Building material (categorical, anonymized).
cond_class – Condition or quality class of the apartment (categorical, anonymized).
has_park – Whether the apartment has a parking space (boolean).
has_balcony – Whether the apartment has a balcony (boolean).
has_lift – Whether the apartment building has an elevator (boolean).
has_sec – Whether the apartment has security features (boolean).
has_store – Whether the apartment has a storage room (boolean).
price_z – Target variable: Apartment price (in appropriate monetary units) to be predicted – only in the training sample
src_month – Source month (time attribute).
loc_code – Anonymized location code of the apartment.
market_volatility – Simulated market fluctuation affecting the apartment price.
infrastructure_quality – Indicator of the building’s infrastructure quality, partially based on the building’s age.
neighborhood_crime_rate – Random index simulating local crime rate.
popularity_index – Randomly generated measure of the apartment’s attractiveness.
green_space_ratio – Proxy variable representing the amount of nearby green space, inversely related to the distance from the city center.
estimated_maintenance_cost – Estimated cost of maintaining the apartment, based on its size.
global_economic_index – Simulated economic index with minor fluctuations across entries, reflecting broader market conditions.

# Files Provided
appartments_train.csv – training data contains 156454 observations and 34 columns along with the target variable price_z.

# Data Processing Method
Numeric & time: KNN imputation
Categorical: add an explicit missing level
Boolean: impute most frequent value
Encoding: one-hot (or hashing if needed)
CV: 5-fold, all preprocessing fitted within fold to avoid leakage