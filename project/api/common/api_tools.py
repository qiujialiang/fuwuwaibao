import os
from pathlib import Path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_dir = os.path.dirname(parent_dir)
sys.path.insert(0, project_dir)

from functools import wraps
import jwt
from flask import request
from project.api.common.constant import LOGIN_SECRET
def token_required():
    def check_token(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            jwt_token=request.headers.get('token',None)
            if not jwt_token:
                return {'error':'token不存在'},401
            user_info=jwt.decode(jwt_token,LOGIN_SECRET,algorithms='HS256')
            if not user_info or not user_info.get('username',None):
                return {'error':'user错误'}
            result=f(*args,**kwargs)
            return result
        return wrapper
    return check_token