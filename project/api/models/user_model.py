from resources import db
from sqlalchemy import Integer,String
from sqlalchemy.orm import Mapped,mapped_column
class UserModel(db.Model):
    __tablename__='users'
    
    id:Mapped[int]=mapped_column(Integer,nullable=False,primary_key=True,autoincrement=True)
    username:Mapped[str]=mapped_column(String(128),nullable=False)
    password:Mapped[str]=mapped_column(String(128),nullable=False)
    
    def serialize(self):
        return {
            'id':self.id,
            'username':self.username,
        }