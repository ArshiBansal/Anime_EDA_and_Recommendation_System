CREATE DATABASE anime_recommendation;
use anime_recommendation;

select* from anime;
select* from rating;

-- 1. Top 10 most popular anime
SELECT name, members FROM anime ORDER BY members DESC LIMIT 10;

-- 2. Top 10 highest-rated anime
SELECT name, rating FROM anime WHERE rating IS NOT NULL ORDER BY rating DESC LIMIT 10;

-- 3. Count of anime by type
SELECT type, COUNT(*) AS anime_count FROM anime GROUP BY type ORDER BY anime_count DESC;

-- 4. Genre-wise average rating
SELECT genre, AVG(rating) AS avg_rating FROM anime WHERE genre IS NOT NULL GROUP BY genre ORDER BY avg_rating DESC;

-- 5. Top 10 most active users
SELECT user_id, COUNT(*) AS total_ratings FROM rating GROUP BY user_id ORDER BY total_ratings DESC LIMIT 10;

-- 6. Average rating per user
SELECT user_id, round(AVG(rating),2) AS avg_user_rating FROM rating WHERE rating IS NOT NULL GROUP BY user_id ORDER BY avg_user_rating DESC;

-- 7. Anime with most ratings
SELECT anime_id, COUNT(*) AS rating_count FROM rating GROUP BY anime_id ORDER BY rating_count DESC LIMIT 10;

-- 8. Most rated anime with names
SELECT a.name, COUNT(r.rating) AS rating_count FROM rating r JOIN anime a ON r.anime_id = a.anime_id GROUP BY a.name ORDER BY rating_count DESC LIMIT 10;

-- 9. Genre-wise rating distribution
SELECT a.genre, AVG(r.rating) AS avg_rating FROM rating r JOIN anime a ON r.anime_id = a.anime_id WHERE r.rating IS NOT NULL GROUP BY a.genre ORDER BY avg_rating DESC;

-- 10. Underrated high-quality anime
SELECT name, rating, members FROM anime WHERE rating > 8 AND members < 10000 ORDER BY rating DESC;

-- 11. Find the average rating by anime type (TV, Movie, etc.)
SELECT type, round(AVG(rating),2) AS avg_rating
FROM anime
WHERE rating IS NOT NULL
GROUP BY type
ORDER BY avg_rating DESC;

-- 12. Find users who rated at least 50 anime
SELECT user_id, COUNT(*) AS rating_count
FROM rating
GROUP BY user_id
HAVING rating_count >= 50
ORDER BY rating_count DESC;

-- 13. List top 5 anime per type by rating
SELECT name, type, rating
FROM (
    SELECT name, type, rating,
           RANK() OVER (PARTITION BY type ORDER BY rating DESC) AS rnk
    FROM anime
    WHERE rating IS NOT NULL
) ranked
WHERE rnk <= 5;

-- 14. Calculate how many anime each user watched but didn't rate (rating = -1)
SELECT user_id, COUNT(*) AS unscored
FROM rating
WHERE rating = -1
GROUP BY user_id
ORDER BY unscored DESC;

-- 15. Find most common anime genres
SELECT genre, COUNT(*) AS genre_count
FROM anime
GROUP BY genre
ORDER BY genre_count DESC;
