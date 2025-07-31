    USE your_database_name;
    CREATE TABLE IF NOT EXISTS newsArticles 
        (id INT AUTO_INCREMENT PRIMARY KEY,
        category	TEXT,
        date	DATETIME,
        title	TEXT,
        topic	TEXT,
        content	TEXT,
        sentiment	DECIMAL,
        likes	INTEGER,
        comments	INTEGER,
        shares	INTEGER,
        keywords	TEXT,
        hashtags	TEXT,
        source_url	TEXT,
        tags	TEXT,
        ctr	REAL,
        spent	INTEGER,
        bounce_rate	REAL,
        location_name	TEXT  );
       