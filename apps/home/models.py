# -*- encoding: utf-8 -*-
from apps import db


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
    return Image.query.filter_by(patient_id=patient_id).order_by(Image.stamp_created.desc()).all()


def request_patient_data(name):
    # name = request.form.get('name')
    patient = query_patient(name)
    if patient is None:
        return (None, None)

    images = query_images(patient.id)
    return (patient, images)


def get_all_patients():
    return Patient.query.all()
