from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'app.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
db = SQLAlchemy(app)



class Patient(db.Model):

    __tablename__ = 'Patient'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(64), unique=True)
    birthdate = db.Column(db.String(64))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                value = value[0]

            setattr(self, property, value)

    def __repr__(self):
        return str(self.name)


class Image(db.Model):

    __tablename__ = 'Image'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer)
    title = db.Column(db.String(64), unique=True)

    filename = db.Column(db.String(64), unique=True)
    cate = db.Column(db.String(64))
    importdate = db.Column(db.String(64))
    importedtime = db.Column(db.String(64))
    stamp_created = db.Column(db.String(64))


    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]

            setattr(self, property, value)

    def __repr__(self):
        return str(self.filename)


def query_patient(name):
    return Patient.query.filter_by(name=name).first()

def query_images(patient_id):
    return Image.query.filter_by(patient_id=patient_id).order_by(Image.importedtime.desc()).all()

def request_patient_data(request):
    name = request.form.get('name')
    patient = query_patient(name)
    if patient is None:
        return (None, None)

    images = query_images(patient.id)
    return (patient, images)



if __name__ == '__main__':
    db.create_all()
    p1 = Patient(id=1000654, name='Tom Huang', birthdate='1955-12-01')
    p2 = Patient(id=2000031,name='Mike Hua', birthdate='1952-09-01')
    p3 = Patient(id=1005000,name='Jesse Ku', birthdate='1975-12-08')
    p4 = Patient(id=3003218,name='Ju Huang', birthdate='1967-04-01')
    p5 = Patient(id=1000178,name='Wang Liu', birthdate='1956-12-04')


    # class 0  Tom Huang
    image11 = Image(patient_id=1000654,title='9001695L',  filename='9001695L.png', cate='L', importdate='2022-01-01', importedtime='12:10:10', stamp_created='2022-01-01 12:10:10')
    image12 = Image(patient_id=1000654,title='9001897L',  filename='9001897L.png', cate='L',importdate='2022-01-01', importedtime='14:10:10', stamp_created='2022-01-01 14:10:10')
    image13 = Image(patient_id=1000654,title='9003126L',  filename='9003126L.png',cate='L', importdate='2022-01-02', importedtime='16:10:10', stamp_created='2022-01-02 16:10:10')
    image14 = Image(patient_id=1000654,title='9003126R',  filename='9003126R.png', cate='R',importdate='2022-01-02', importedtime='12:11:10', stamp_created='2022-01-02 12:11:10')
    image15 = Image(patient_id=1000654,title='9003430L', filename='9003430L.png', cate='L',importdate='2022-01-02', importedtime='12:10:16', stamp_created='2022-01-02 12:10:16')
    image16 = Image(patient_id=1000654,title='9004315L',  filename='9004315L.png',cate='L', importdate='2022-01-02', importedtime='14:10:16', stamp_created='2022-01-02 14:10:16')
    image17 = Image(patient_id=1000654, title='9022703R', filename='9022703R.png', cate='R', importdate='2022-01-05',importedtime='14:10:16', stamp_created='2022-01-05 14:10:16')



    # class 4  Mike Hua
    image21 = Image(patient_id=2000031,title='9031426R',  filename='9031426R.png', cate='R', importdate='2022-01-01', importedtime='12:10:10', stamp_created='2022-01-01 12:10:10')
    image22 = Image(patient_id=2000031,title='9025994L',  filename='9025994L.png', cate='L',importdate='2022-01-01', importedtime='14:10:10', stamp_created='2022-01-01 14:10:10')
    image23 = Image(patient_id=2000031,title='9156694R',  filename='9156694R.png',cate='R', importdate='2022-01-02', importedtime='16:10:10', stamp_created='2022-01-02 16:10:10')
    image24 = Image(patient_id=2000031,title='9039627L',  filename='9039627L.png', cate='L',importdate='2022-01-02', importedtime='12:11:10', stamp_created='2022-01-02 12:11:10')
    image25 = Image(patient_id=2000031,title='9049507L', filename='9049507L.png', cate='L',importdate='2022-01-02', importedtime='12:10:16', stamp_created='2022-01-02 12:10:16')
    image26 = Image(patient_id=2000031,title='9118430R',  filename='9118430R.png',cate='R', importdate='2022-01-03', importedtime='14:10:16', stamp_created='2022-01-03 14:10:16')
    image27 = Image(patient_id=2000031,title='9413071R',  filename='9413071R.png',cate='R', importdate='2022-01-07', importedtime='14:10:16', stamp_created='2022-01-07 14:10:16')


    # class 1  Jesse Ku
    image31 = Image(patient_id=1005000,title='9014797R',  filename='9014797R.png', cate='R', importdate='2022-01-01', importedtime='12:10:10', stamp_created='2022-01-01 12:10:10')
    image32 = Image(patient_id=1005000,title='9002316L',  filename='9002316L.png', cate='L',importdate='2022-01-01', importedtime='14:10:10', stamp_created='2022-01-01 14:10:10')
    image33 = Image(patient_id=1005000,title='9002316R',  filename='9002316R.png',cate='R', importdate='2022-01-02', importedtime='16:10:10', stamp_created='2022-01-02 16:10:10')
    image34 = Image(patient_id=1005000,title='9004175L',  filename='9004175L.png', cate='L',importdate='2022-01-02', importedtime='12:11:10', stamp_created='2022-01-02 12:11:10')
    image35 = Image(patient_id=1005000,title='9008561L', filename='9008561L.png', cate='L',importdate='2022-01-02', importedtime='12:10:16', stamp_created='2022-01-02 12:10:16')
    image36 = Image(patient_id=1005000,title='9003430R',  filename='9003430R.png',cate='R', importdate='2022-01-03', importedtime='14:10:16', stamp_created='2022-01-03 14:10:16')
    image37 = Image(patient_id=1005000,title='9057150R',  filename='9057150R.png',cate='R', importdate='2022-01-07', importedtime='14:10:16', stamp_created='2022-01-07 14:10:16')


    # class 2  Ju Huang
    image41 = Image(patient_id=3003218,title='9010370R',  filename='9010370R.png', cate='R', importdate='2022-01-01', importedtime='12:10:10', stamp_created='2022-01-01 12:10:10')
    image42 = Image(patient_id=3003218,title='9011420L',  filename='9011420L.png', cate='L',importdate='2022-01-01', importedtime='14:10:10', stamp_created='2022-01-01 14:10:10')
    image43 = Image(patient_id=3003218,title='9011420R',  filename='9011420R.png',cate='R', importdate='2022-01-02', importedtime='16:10:10', stamp_created='2022-01-02 16:10:10')
    image44 = Image(patient_id=3003218,title='9013798L',  filename='9013798L.png', cate='L',importdate='2022-01-02', importedtime='12:11:10', stamp_created='2022-01-02 12:11:10')
    image45 = Image(patient_id=3003218,title='9024940L', filename='9024940L.png', cate='L',importdate='2022-01-02', importedtime='12:10:16', stamp_created='2022-01-02 12:10:16')
    image46 = Image(patient_id=3003218,title='9015798R',  filename='9015798R.png',cate='R', importdate='2022-01-03', importedtime='14:10:16', stamp_created='2022-01-03 14:10:16')
    image47 = Image(patient_id=3003218,title='9044902R',  filename='9044902R.png',cate='R', importdate='2022-01-07', importedtime='14:10:16', stamp_created='2022-01-07 14:10:16')


    # class 3  Wang Liu
    image51 = Image(patient_id=1000178,title='9001104R',  filename='9001104R.png', cate='R', importdate='2022-01-01', importedtime='12:10:10', stamp_created='2022-01-01 12:10:10')
    image52 = Image(patient_id=1000178,title='9000099L',  filename='9000099L.png', cate='L',importdate='2022-01-01', importedtime='14:10:10', stamp_created='2022-01-01 14:10:10')
    image53 = Image(patient_id=1000178,title='9001897R',  filename='9001897R.png',cate='R', importdate='2022-01-02', importedtime='16:10:10', stamp_created='2022-01-02 16:10:10')
    image54 = Image(patient_id=1000178,title='9002430L',  filename='9002430L.png', cate='L',importdate='2022-01-02', importedtime='12:11:10', stamp_created='2022-01-02 12:11:10')
    image55 = Image(patient_id=1000178,title='9014797L', filename='9014797L.png', cate='L',importdate='2022-01-02', importedtime='12:10:16', stamp_created='2022-01-02 12:10:16')
    image56 = Image(patient_id=1000178,title='9013161R',  filename='9013161R.png',cate='R', importdate='2022-01-03', importedtime='14:10:16', stamp_created='2022-01-03 14:10:16')
    image57 = Image(patient_id=1000178,title='9058146R',  filename='9058146R.png',cate='R', importdate='2022-01-07', importedtime='14:10:16', stamp_created='2022-01-07 14:10:16')


    db.session.add(p1)
    db.session.add(p2)
    db.session.add(p3)
    db.session.add(p4)
    db.session.add(p5)

    db.session.add(image11)
    db.session.add(image12)
    db.session.add(image13)
    db.session.add(image14)
    db.session.add(image15)
    db.session.add(image16)
    db.session.add(image17)


    db.session.add(image21)
    db.session.add(image22)
    db.session.add(image23)
    db.session.add(image24)
    db.session.add(image25)
    db.session.add(image26)
    db.session.add(image27)

    db.session.add(image31)
    db.session.add(image32)
    db.session.add(image33)
    db.session.add(image34)
    db.session.add(image35)
    db.session.add(image36)
    db.session.add(image37)

    db.session.add(image41)
    db.session.add(image42)
    db.session.add(image43)
    db.session.add(image44)
    db.session.add(image45)
    db.session.add(image46)
    db.session.add(image47)

    db.session.add(image51)
    db.session.add(image52)
    db.session.add(image53)
    db.session.add(image54)
    db.session.add(image55)
    db.session.add(image56)
    db.session.add(image57)


    db.session.commit()






