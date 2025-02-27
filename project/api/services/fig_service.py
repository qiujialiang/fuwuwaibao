from models.fig_model import FigModel
from sqlalchemy import Select
from resources import db

class FigService:
    def get_fig_by_name(self,name:str):
        query=Select(FigModel).where(FigModel.Name==name)
        return db.session.scalars(query).first()
    def creat_fig(self,fig_model:FigModel):
        exist_fig=self.get_fig_by_name(fig_model.Name)
        if exist_fig:
            self.update_fig(fig_model)
        else:
            db.session.add(fig_model)
            db.session.commit()
            return fig_model
    def update_fig(self,fig_model:FigModel):
        exist_fig=self.get_fig_by_name(fig_model.Name)
        if not exist_fig:
            raise Exception(f"not found")
        exist_fig.Name=fig_model.Name
        exist_fig.Raw_Fig=fig_model.Raw_Fig
        exist_fig.Res_class=fig_model.Res_class
        exist_fig.Res_seg=fig_model.Res_seg
        exist_fig.Seg_fig=fig_model.Seg_Fig
        exist_fig.Time=fig_model.Time
        
        db.session.commit()
        return exist_fig