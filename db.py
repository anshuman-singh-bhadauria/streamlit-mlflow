class Database:
    def __init__(self, name, host, port, user, pwd):
        self.name = name
        self.host = host
        self.port = port
        self.user = user
        self.pwd = pwd


class Table:
    def __init__(self, table_name,db_name, host, port, user, pwd):
        self.table_name = table_name
        self.db_name = db_name
        self.host = host
        self.port = port
        self.user = user
        self.pwd = pwd
