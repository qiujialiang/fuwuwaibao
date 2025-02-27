from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restx import Api

app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='mysql+mysqldb://root:12345@127.0.0.1/project'
db=SQLAlchemy(app)
api = Api(app, version='1.0', title='API Documentation', description='项目的Api接口文档')

