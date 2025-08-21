-- How many food providers and receivers are there in each city?
WITH num_providers as(
SELECT
  provider_city AS City,
  COUNT(DISTINCT provider_id) AS NumberOfProviders
FROM merged_data
GROUP BY provider_city
ORDER BY COUNT(DISTINCT provider_id) DESC),

num_receivers as (SELECT
  receiver_city AS City,
  COUNT(DISTINCT receiver_id) AS NumberOfReceivers
FROM merged_data 
GROUP BY receiver_city
ORDER BY COUNT(DISTINCT receiver_id) DESC)

SELECT
	num_receivers.City, NumberOfProviders, NumberOfReceivers
FROM
	num_receivers ,
	num_providers
WHERE num_receivers.City = num_providers.City;


-- Which type of food provider (restaurant, grocery store, etc.) contributes the most food?

SELECT
  provider_type,
  SUM(quantity) AS TotalFoodContributed
FROM merged_data
GROUP BY
  provider_type
ORDER BY
  TotalFoodContributed DESC;
  
  
--  What is the contact information of food providers in a specific city?

SELECT DISTINCT
  provider_name,
  provider_contact
FROM merged_data
WHERE
  provider_city = '[CityName]';
  
  
--  Which receivers have claimed the most food?

SELECT
  receiver_name,
  SUM(quantity) AS TotalFoodClaimed
FROM merged_data
WHERE
  status = 'Completed'
GROUP BY
  receiver_name
ORDER BY
  TotalFoodClaimed DESC;
  
  
--  What is the total quantity of food available from all providers?

SELECT
  SUM(quantity) AS TotalFoodQuantity
FROM merged_data;


-- Which city has the highest number of food listings?

SELECT
  provider_city,
  COUNT(DISTINCT food_id) AS NumberOfListings
FROM merged_data
GROUP BY
  provider_city
ORDER BY
  NumberOfListings DESC
LIMIT 1;


-- What are the most commonly available food types?

SELECT
  food_type,
  COUNT(*) AS NumberOfListings
FROM merged_data
GROUP BY
  food_type
ORDER BY
  NumberOfListings DESC;
  
  
--  How many food claims have been made for each food item?

SELECT
  food_name,
  COUNT(claim_id) AS NumberOfClaims
FROM merged_data
GROUP BY
  food_name
ORDER BY
  NumberOfClaims DESC;
  

--  Which provider has had the highest number of successful food claims?

SELECT
  provider_name,
  COUNT(claim_id) AS NumberOfSuccessfulClaims
FROM merged_data
WHERE
  status = 'Completed'
GROUP BY
  provider_name
ORDER BY
  NumberOfSuccessfulClaims DESC
LIMIT 1;


--  What percentage of food claims are completed vs. pending vs. canceled?

SELECT
  status,
  COUNT(*) AS NumberOfClaims,
  (COUNT(*) * 100.0 / (
    SELECT
      COUNT(*)
    FROM merged_data
  )) AS Percentage
FROM merged_data
GROUP BY
  status;
  
--  What is the average quantity of food claimed per receiver?

SELECT
  receiver_name,
  AVG(quantity) AS AverageQuantityClaimed
FROM merged_data
WHERE
  status = 'Completed'
GROUP BY
  receiver_name
ORDER BY
  AverageQuantityClaimed DESC;
  
  
-- Which meal type (breakfast, lunch, dinner, snacks) is claimed the most?

SELECT
  meal_type,
  COUNT(claim_id) AS NumberOfClaims
FROM merged_data
WHERE
  status = 'Completed'
GROUP BY
  meal_type
ORDER BY
  NumberOfClaims DESC
LIMIT 1;


-- What is the total quantity of food donated by each provider?

SELECT
  provider_name,
  SUM(quantity) AS TotalQuantityDonated
FROM merged_data
GROUP BY
  provider_name
ORDER BY
  TotalQuantityDonated DESC;