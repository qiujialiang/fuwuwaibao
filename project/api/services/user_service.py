from models.user_model import UserModel
from sqlalchemy import Select
from resources import db

class UserService:
    def login(self,username:str,password:str):
        query=Select(UserModel).where(UserModel.username==username)
        user_model=db.session.scalars(query).first()
        if user_model and user_model.password==password:
            return user_model
        else:
            return None
        
    def register(self,username:str,password:str):
        query=Select(UserModel).where(UserModel.username==username)
        if query is not None:
            user_model=UserModel(username=username,password=password)
            db.session.add(user_model)
            db.session.commit()
            return user_model