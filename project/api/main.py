import warnings
from flask import Flask
from flask_restx import Api
from resources.fig_resources import api as fig_api
from resources.user_resources import api as user_api
from resources import app,api

warnings.filterwarnings('ignore')

api.add_namespace(user_api)
api.add_namespace(fig_api)

if __name__=="__main__":
    app.run(debug=True,port=5000)