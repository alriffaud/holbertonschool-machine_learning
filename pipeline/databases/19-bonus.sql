-- This script that creates a stored procedure AddBonus that adds a new correction for a student.
-- Requirements:
-- Procedure AddBonus is taking 3 inputs (in this order):
--  user_id, a users.id value (you can assume user_id is linked to an existing users)
--  project_name, a new or already exists projects - if no projects.name found in the table, you should create it
--  score, the score value for the correction

DELIMITER //
CREATE PROCEDURE AddBonus(
    IN user_id INT,
    IN project_name VARCHAR(255),
    IN score INT
)
BEGIN
    DECLARE proj_id INT DEFAULT NULL;
    -- If no project is found, set proj_id to NULL without throwing an error
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET proj_id = NULL;

    -- Check if the project already exists in the projects table
    SELECT id INTO proj_id
    FROM projects
    WHERE name = project_name
    LIMIT 1;

    -- If the project does not exist, insert it and get the new project id
    IF proj_id IS NULL THEN
        INSERT INTO projects (name) VALUES (project_name);
        SET proj_id = LAST_INSERT_ID();
    END IF;

    -- Insert a new correction for the student with the given score
    INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, proj_id, score);
END;
//

DELIMITER ;
