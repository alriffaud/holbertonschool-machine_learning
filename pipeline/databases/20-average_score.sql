-- This script creates a stored procedure ComputeAverageScoreForUser that computes
-- and store the average score for a student.
-- Requirements:
-- Procedure ComputeAverageScoreForUser is taking 1 input:
--  user_id, a users.id value (you can assume user_id is linked to an existing users)

DELIMITER //

DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser //

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    UPDATE users
    -- Calculate the average score for the given user_id
    SET average_score = (
        SELECT AVG(score)
        FROM corrections
        WHERE corrections.user_id = user_id
    )
    WHERE id = user_id;
END;
//

DELIMITER ;
