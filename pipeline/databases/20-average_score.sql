-- This script creates a stored procedure ComputeAverageScoreForUser that computes
-- and store the average score for a student.
-- Requirements:
-- Procedure ComputeAverageScoreForUser is taking 1 input:
--  user_id, a users.id value (you can assume user_id is linked to an existing users)

DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE avgScore FLOAT;
    
    -- Calculate the average score for the given user_id
    SELECT IFNULL(AVG(score), 0) INTO avgScore
    FROM corrections
    WHERE user_id = user_id;
    
    -- Update the average_score in the users table for the given user
    UPDATE users
    SET average_score = avgScore
    WHERE id = user_id;
END;
//

DELIMITER ;
