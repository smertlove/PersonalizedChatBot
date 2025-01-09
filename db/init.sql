DROP DATABASE IF EXISTS `ChatbotMemoryDB`;
CREATE DATABASE `ChatbotMemoryDB`        ;
USE `ChatbotMemoryDB`                    ;


SET NAMES UTF8                            ;
SET CHARACTER SET UTF8                    ;
SET collation_connection='utf8_general_ci';
SET collation_database='utf8_general_ci'  ;
SET collation_server='utf8_general_ci'    ;
SET character_set_client='utf8'           ;
SET character_set_connection='utf8'       ;
SET character_set_database='utf8'         ;
SET character_set_results='utf8'          ;
SET character_set_server='utf8'           ;



-- Планировалось хранить такие триплеты, но Вы можете предложить свой вариант
CREATE TABLE IF NOT EXISTS Memory (
    id         INT          PRIMARY KEY AUTO_INCREMENT NOT NULL,
    added_at   DATETIME     DEFAULT CURRENT_TIMESTAMP  NOT NULL,
    subj       varCHAR(255)                                    ,
    pred       varCHAR(255)                                    ,
    obj        varCHAR(255)                                    ,
    other_info TEXT
);


/*
    NOTE:
    База создается на старте контейнера, но только один раз
    (она сохранится в памяти, и при повторном пуске код не отработает).
    Поэтому не забывайте делать `sudo docker container prune`, если будете вносить правки в базу.
*/



CREATE USER 'query_retriever'@'%' IDENTIFIED BY 'qwerty';
GRANT SELECT, INSERT, UPDATE, DELETE ON ChatbotMemoryDB.* TO 'query_retriever'@'%';

CREATE USER 'db_retriever'@'%' IDENTIFIED BY 'qwerty';
GRANT SELECT, INSERT, UPDATE, DELETE ON ChatbotMemoryDB.* TO 'db_retriever'@'%';

FLUSH PRIVILEGES;




