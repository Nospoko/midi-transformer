import os

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
DB_DSN = "postgresql://localhost/midi_transformers?user=root&password=my_password"


class DatabaseConnection:
    def __init__(self):
        db_url = sa.engine.make_url(DB_DSN)
        # Pre-ping allows the connection to stay alive in long running sessions with low activity
        # https://docs.sqlalchemy.org/en/20/core/pooling.html#sqlalchemy.pool.Pool.params.pre_ping
        self.__engine = sa.create_engine(db_url, pool_pre_ping=True)

    @property
    def engine(self) -> sa.engine.Engine:
        return self.__engine

    def read_df(self, query: str) -> pd.DataFrame:
        df = pd.read_sql(
            sql=query,
            con=self.__engine,
        )
        return df

    def execute(self, query: str):
        with self.__engine.connect() as connection:
            result = connection.execute(sa.text(query))
            connection.commit()

        return result

    def read_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        with self.__engine.connect() as connection:
            df = pd.read_sql(
                sql=sql,
                con=connection,
                **kwargs,
            )
        return df

    def to_sql(self, df: pd.DataFrame, table: str, **kwargs):
        with self.__engine.connect() as connection:
            df.to_sql(
                table,
                con=connection,
                **kwargs,
            )


database_cnx = DatabaseConnection()
