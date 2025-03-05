import pymysql


class DatabaseHelper:
    def __init__(self, password, database="steel_defect", host="localhost", user="root", port=3306):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        try:
            self.conn = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port,
            )
            self.cursor = self.conn.cursor()
            print(f"成功连接到数据库: {self.database}")
        except pymysql.MySQLError as e:
            print(f"数据库连接失败: {e}")
            return

        self.init_database()

    def save_result(self, result):
        """Insert result to database."""
        insert_query = """
        INSERT INTO res (name, raw_fig, res_fig, date, time, label, num, dice)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        # Unpack result properties into a tuple
        result_data = (result.name, result.raw_fig, result.res_fig, result.date, result.time, result.label, result.num, result.dice)
        self.cursor.execute(insert_query, result_data)
        self.conn.commit()

    def init_database(self):
        """Initialize database and table if not exists."""
        try:
            # Create the database if it does not exist
            create_db_query = f"CREATE DATABASE IF NOT EXISTS {self.database} DEFAULT CHARSET=utf8mb4"
            self.cursor.execute(create_db_query)
            self.conn.select_db(self.database)  # Switch to the database

            # Create the table if it does not exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS `res` (
                `name` VARCHAR(255) NOT NULL PRIMARY KEY,
                `raw_fig` MEDIUMBLOB NOT NULL,
                `res_fig` MEDIUMBLOB NOT NULL,
                `date` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `time` VARCHAR(255) NOT NULL,
                `label` VARCHAR(255) NOT NULL,
                `num` VARCHAR(255) NOT NULL,
                `dice` VARCHAR(255) NOT NULL
            )
            """
            self.cursor.execute(create_table_query)
            self.conn.commit()
            print("数据库初始化完成")

        except pymysql.MySQLError as e:
            print(f"数据库初始化失败：{e}")

    def truncate_database(self):
        truncate_query = "TRUNCATE `res`"
        self.cursor.execute(truncate_query)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("数据库连接已关闭")


class DetectResult:
    def __init__(self, name, raw_fig, res_fig, date, time, label, num, dice):
        self.name = name
        self.raw_fig = raw_fig
        self.res_fig = res_fig
        self.date = date
        self.time = time
        self.label = label
        self.num = num
        self.dice = dice


class DetectObj:
    def __init__(self, name, figure, path=None):
        self.name = name
        self.figure = figure
        self.path = path
