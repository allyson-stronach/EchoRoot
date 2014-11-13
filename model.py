#when i need it: SQLAlchemy tutorial http://docs.sqlalchemy.org/en/rel_0_9/orm/tutorial.html
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, backref

Base = declarative_base()

# THIS IS THE FUCKING BEST http://zetcode.com/db/mysqlpython/

#found this at: http://docs.sqlalchemy.org/en/rel_0_9/dialects/mysql.html#module-sqlalchemy.dialects.mysql.mysqldb
#also, this http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html
engine = create_engine("mysql+mysqldb://root@localhost:3306/backpageads?charset=utf8&use_unicode=0")
session = scoped_session(sessionmaker(bind=engine,
                                      autocommit = False,
                                      autoflush = False))


#used this page to check SQLAlchemy column and data types http://docs.sqlalchemy.org/en/latest/core/types.html
class Ad(Base):
    __tablename__ = "ads"

    id = Column(Integer, primary_key = True)
    first_id = Column(Integer, nullable=True)
    sources_id = Column(Integer, nullable=False)
    incoming_id = Column(Integer, nullable=False)
    url = Column(String(2083), nullable=False)
    title = Column(String(1024), nullable=False)
    text = Column(String, nullable=False)
    type = Column(String(16), nullable=True)
    sid = Column(String(64), nullable=True)
    region = Column(String(128), nullable=True) 
    city = Column(String(128), nullable=True)   
    state = Column(String(64), nullable=True)
    country = Column(String(64), nullable=True)
    phone = Column(String(64), nullable=True)
    age = Column(String(10), nullable=True)
    website = Column(String(2048), nullable=True)
    email = Column(String(512), nullable=True)
    gender = Column(String(20), nullable=True)
    service = Column(String(16), nullable=True)
    posttime = Column(DateTime, nullable=True)
    importtime = Column(DateTime, nullable=False)
    modtime = Column(DateTime, nullable=False)

    #ads table is backreferenced from ads_attributes table

class AdAttribute(Base):
    __tablename__ = "ads_attributes"

    id = Column(Integer, primary_key = True)
    ads_id = Column(Integer, ForeignKey('ads.id'), nullable=False)
    attribute = Column(String(32), nullable=False)
    value = Column(String(2500), nullable=False)
    extracted = Column(Integer, nullable=False)
    extractedraw = Column(String(512), nullable=True)
    modtime = Column(DateTime, nullable=False)

    ads = relationship("Ad", backref=backref("ads_attributes", order_by=id))


def main():
    """In case we need this for something"""
    pass

if __name__ == "__main__":
    main()



