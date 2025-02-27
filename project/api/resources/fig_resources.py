import io
import os
from pathlib import Path
import sys

import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_dir = os.path.dirname(parent_dir)
sys.path.insert(0, project_dir)

from datetime import datetime
from flask import send_file
from flask_restx import Resource, reqparse, Namespace

from werkzeug.datastructures import FileStorage
from project.api.common.api_tools import token_required
from project.utilis.eda import eda
from resources import api
from models.fig_model import FigModel
from services.fig_service import FigService
from project.utilis.pred import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_class = r"models\class\model_class_final 0.85.pth"
model_class = torch.load(path_class, map_location=device)
path_seg = r"models\seg\model_FPN_final 0.899.pth"
model_seg = torch.load(path_seg,map_location=device)

api = Namespace('detect', description='Steel defect detection API')
@api.route('/')
class FigResource(Resource):
    @token_required()
    @api.doc(responses={200: 'Success', 400: 'Validation Error'}, params={'fig': '上传需要检测的图片，返回检测结果并保存至数据库'})
    def post(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('fig', required=True, type=FileStorage, location='files', help="error file")
        
        args=self.parser.parse_args()
        attach=args.get('fig')
        name=attach.filename
        image_data=attach.read()
        time=datetime.now()
        
        probability_label,df_segmentation,_=predict(name,image_data,mode=2,model_class=model_class,model_seg=model_seg)
        res_class=str(probability_label['probability_label'].values)
        res_seg=str(df_segmentation['EncodedPixels'].values)
        fig_data=eda(df_segmentation,image_data)
        
        #预测
        fig_model=FigModel(Name=name,Raw_Fig=image_data,Res_class=res_class,Res_seg=res_seg,Seg_Fig=fig_data,Time=time)
        FigService().creat_fig(fig_model)
        stream=io.BytesIO(fig_data)
        return send_file(stream,mimetype='image/jpeg',download_name=name)
@api.route('/<string:filename>')
class Fig(Resource):
    @token_required()
    @api.doc(responses={200: 'Success', 400: 'Validation Error'}, params={'fig': '输入图像名称，返回该图像的检测结果'})
    def get(self,filename):
        fig_model=FigService().get_fig_by_name(name=filename)
        if fig_model:
            return fig_model.serialize()
        else:
            return {"未找出数据"}
