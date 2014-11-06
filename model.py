#when i need it: SQLAlchemy tutorial http://docs.sqlalchemy.org/en/rel_0_9/orm/tutorial.html
import MySQLdb as mdb
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref

Base = declarative_base()

# THIS IS THE FUCKING BEST http://zetcode.com/db/mysqlpython/

#found this at: http://docs.sqlalchemy.org/en/rel_0_9/dialects/mysql.html#module-sqlalchemy.dialects.mysql.mysqldb
#also, this http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html
engine = create_engine("mysql+mysqldb://user:pass@host/dbname?charset=utf8&use_unicode=0")
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






# from http://zetcode.com/db/mysqlpython/
# #From the connection, we get the cursor object. The cursor is used to traverse the records from the result set. 
# con = mdb.connect('localhost', 'allysonstronach', '', 'backpageads')
# #We call the execute() method of the cursor and execute the SQL statement.
# cur = con.cursor()
# cur.execute("SELECT * FROM ads limit 1")

# #We fetch the data. Since we retrieve only one record (as written above), we call the fetchone() method.
# ver = cur.fetchone()


# #We print the data that we have retrieved to the console.
# print "Database version : %s " % ver

# #We check for errors. This is important, since working with databases is error prone.
# except mdb.Error, e:
  
#     print "Error %d: %s" % (e.args[0],e.args[1])
#     sys.exit(1)

# # In the final step, we release the resources.
# $ ./version.py
# Database version : 5.5.9 






# used this site: http://www.slideshare.net/SarahGuido/a-beginners-guide-to-machine-learning-with-scikitlearn

# from sklearn import cluster

# #create clustering object 
# k_means = cluster.KMeans(n_clusters=4)

# #fit the clustering model on the data
# KM = k_means.fit(blah_data)



