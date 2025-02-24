import io
import os
from pathlib import Path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_dir = os.path.dirname(parent_dir)
sys.path.insert(0, project_dir)

from datetime import datetime
from flask import request, send_file
from flask_restful import Resource,reqparse
from flask_restx import Resource, reqparse

from werkzeug.datastructures import FileStorage
from project.api.common.api_tools import token_required
from project.utilis.eda import eda
from resources import api
from models.fig_model import FigModel
from services.fig_service import FigService
from project.utilis.pred import predict
class FigResource(Resource):
    def __init__(self):
        self.parser=reqparse.RequestParser()
        self.parser.add_argument('fig',required=True,type=FileStorage,location='files',help="error file")
    @token_required()
    @api.doc(responses={200: 'Success', 400: 'Validation Error'}, params={'fig': '上传需要检测的图片，返回检测结果并保存至数据库'})
    def post(self):
        attach=self.parser.parse_args().get('fig')
        name=attach.filename
        # base_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent  # 获取项目根目录
        # input_path = base_path.joinpath('api', 'attachment', 'input', name)
        # output_path = base_path.joinpath('api', 'attachment', 'output', name)
        image_data=attach.read()
        # attach.seek(0)
        # attach.save(input_path)
        time=datetime.now()
        
        probability_label,df_segmentation=predict(name,image_data,mode=3,)
        res_class=str(probability_label['probability_label'].values)
        res_seg=str(df_segmentation['EncodedPixels'].values)
        fig_data=eda(df_segmentation,image_data)
        
        #预测
        #file_path=Path(r'project\api\attachment\output').joinpath(name)
        fig_model=FigModel(Name=name,Raw_Fig=image_data,Res_class=res_class,Res_seg=res_seg,Seg_Fig=fig_data,Time=time)
        FigService().creat_fig(fig_model)
        stream=io.BytesIO(fig_data)
        # image=Image.open(stream)
        # response.headers['Data'] = fig_model.serialize()
        return send_file(stream,mimetype='image/jpeg',download_name=name)
class Fig(Resource):
    @token_required()
    @api.doc(responses={200: 'Success', 400: 'Validation Error'}, params={'fig': '输入图像名称，返回该图像的检测结果'})
    def get(self,filename):
        fig_model=FigService().get_fig_by_name(name=filename)
        if fig_model:
            return fig_model.serialize()
        else:
            return {"未找出数据"}
api.add_resource(FigResource,'/fig')
api.add_resource(Fig,'/fig/<filename>')