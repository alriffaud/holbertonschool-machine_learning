-- this script lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.
-- Each record display: tv_shows.title - tv_show_genres.genre_id
-- Results is sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
-- Only one SELECT statement can be used

SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
INNER JOIN tv_show_genres
ON tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title, tv_show_genres.genre_id;
