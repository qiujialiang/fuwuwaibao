from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from flask_restx import Namespace

app=Flask(__name__)
api=Api(app)
api=Namespace('缺陷检测api', description='包括图像上传检测、查询结果、token获取')
app.config['SQLALCHEMY_DATABASE_URI']='mysql+mysqldb://root:12345@127.0.0.1/project'
db=SQLAlchemy(app)

from resources import fig_resources
from resources import user_resources