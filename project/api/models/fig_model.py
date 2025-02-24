from resources import db
from datetime import datetime
from sqlalchemy import String,TIMESTAMP,LargeBinary
from sqlalchemy.orm import Mapped,mapped_column
class FigModel(db.Model):
    __tablename__='res_from_api'
    Name:Mapped[str]=mapped_column(String(100),nullable=False,primary_key=True)
    Raw_Fig:Mapped[bytes]=mapped_column(LargeBinary,nullable=False)
    Res_class:Mapped[str]=mapped_column(String(100),nullable=False)
    Res_seg:Mapped[str]=mapped_column(String(512),nullable=False)
    Seg_Fig:Mapped[bytes]=mapped_column(LargeBinary,nullable=False)
    Time:Mapped[datetime]=mapped_column(TIMESTAMP,nullable=False)
    
    def serialize(self):
        return {
            'Name':self.Name,
            'Time':self.Time.isoformat(),
            'Res_class':self.Res_class,
            'Res_seg':self.Res_seg,
            'Seg_Fig':str(self.Seg_Fig),
        }