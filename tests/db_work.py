from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir,'../apps', 'app.db')
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


    # class 0
    image11 = Image(patient_id=1000654,title='9001695L',  filename='9001695L.png', cate='L', importdate='2022-01-01', importedtime='12:10:10', stamp_created='2022-01-01 12:10:10')
    image12 = Image(patient_id=1000654,title='9001897L',  filename='9001897L.png', cate='L',importdate='2022-01-01', importedtime='14:10:10', stamp_created='2022-01-01 14:10:10')
    image13 = Image(patient_id=1000654,title='9003126L',  filename='9003126L.png',cate='L', importdate='2022-01-02', importedtime='16:10:10', stamp_created='2022-01-02 16:10:10')
    image14 = Image(patient_id=1000654,title='9003126R',  filename='9003126R.png', cate='R',importdate='2022-01-02', importedtime='12:11:10', stamp_created='2022-01-02 12:11:10')
    image15 = Image(patient_id=1000654,title='9003430L', filename='9003430L.png', cate='L',importdate='2022-01-02', importedtime='12:10:16', stamp_created='2022-01-02 12:10:16')
    image16 = Image(patient_id=1000654,title='9004315L',  filename='9004315L.png',cate='L', importdate='2022-01-02', importedtime='14:10:16', stamp_created='2022-01-02 14:10:16')
    # class 1
    image1 = Image(patient_id=2000031,title='9007904R',  filename='9007904R.png', cate='R', importdate='2022-01-01', importedtime='12:10:10', stamp_created='2022-01-01 12:10:10')
    image2 = Image(patient_id=2000031,title='9008820L',  filename='9008820L.png', cate='L',importdate='2022-01-01', importedtime='14:10:10', stamp_created='2022-01-01 14:10:10')
    image3 = Image(patient_id=2000031,title='9008820R',  filename='9008820R.png',cate='R', importdate='2022-01-02', importedtime='16:10:10', stamp_created='2022-01-02 16:10:10')
    image4 = Image(patient_id=2000031,title='9009623L',  filename='9009623L.png', cate='L',importdate='2022-01-02', importedtime='12:11:10', stamp_created='2022-01-02 12:11:10')
    image5 = Image(patient_id=2000031,title='9010370L', filename='9010370L.png', cate='L',importdate='2022-01-02', importedtime='12:10:16', stamp_created='2022-01-02 12:10:16')
    image6 = Image(patient_id=2000031,title='9010952L',  filename='9010952L.png',cate='L', importdate='2022-01-02', importedtime='14:10:16', stamp_created='2022-01-02 14:10:16')
    image7 = Image(patient_id=2000031,title='9010952R',  filename='9010952R.png',cate='R', importdate='2022-01-03', importedtime='14:10:16', stamp_created='2022-01-03 14:10:16')



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


    db.session.add(image1)
    db.session.add(image2)
    db.session.add(image3)
    db.session.add(image4)
    db.session.add(image5)
    db.session.add(image6)
    db.session.add(image7)



    db.session.commit()






