-- This script displays the max temperature of each state (ordered by State name)
-- of the database hbtn_0c_0 in a MySQL server.

SELECT state, MAX(value) AS max_temp FROM temperatures
GROUP BY state
ORDER BY state;
