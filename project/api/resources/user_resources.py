import os
from pathlib import Path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from flask_restful import Resource
from api.common.constant import LOGIN_SECRET
from api.resources import api
from services.user_service import UserService
from flask import request
import jwt
class LoginResource(Resource):
    @api.doc(responses={200: 'Success', 400: 'Validation Error'}, params={'user': '用户登录'})
    def post(self):
        request_json = request.json
        if request_json:
            username = request_json.get('username', None)
            password = request_json.get('password', None)
            user_model = UserService().login(username, password)
            if user_model:
                user_json = user_model.serialize()
                jwt_token=jwt.encode(user_json,LOGIN_SECRET,algorithm='HS256')
                user_json['token']=jwt_token
                return user_json
            else:
                return {'error': 'error'}
class RegisterResource(Resource):
    @api.doc(responses={200: 'Success', 400: 'Validation Error'}, params={'user': '注册用户，返回token'})
    def post(self):
        request_json=request.json
        if request_json:
            username=request_json.get('username',None)
            password=request_json.get('password',None)
            user_model=UserService().register(username,password)
            
            if user_model:
                user_json = user_model.serialize()
                jwt_token=jwt.encode(user_json,LOGIN_SECRET,algorithm='HS256')
                user_json['token']=jwt_token
                return user_json
            else:
                return {'error': 'error'}
api.add_resource(LoginResource,'/login')
api.add_resource(RegisterResource,'/register')