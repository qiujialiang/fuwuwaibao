from flask import Flask
from flask_restx import Api
from resources import api as rest_api

app = Flask(__name__)
api = Api(app, version='1.0', title='API Documentation',
          description='项目的Api接口文档')
api.add_namespace(rest_api)
if __name__=="__main__":
    app.run(debug=True,port=5000)