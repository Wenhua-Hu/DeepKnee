from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField


class SearchForm(FlaskForm):
    search_name = StringField('', id='search_name')
    submit = SubmitField('Search', id="search_button")
