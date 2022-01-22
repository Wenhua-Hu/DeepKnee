# -*- encoding: utf-8 -*-
from sys import exit

from decouple import config
from flask_migrate import Migrate

from apps import create_app, db
from apps.config import config_dict

# WARNING: Don't run with debug turned on in production!
DEBUG = config('DEBUG', default=False, cast=bool)


# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'


try:

    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)

Migrate(app, db)

if DEBUG:
    app.logger.info('DEBUG       = ' + str(DEBUG))
    app.logger.info('Environment = ' + get_config_mode)
    app.logger.info('DBMS        = ' + app_config.SQLALCHEMY_DATABASE_URI)

if __name__ == "__main__":
    print("--------- run from terminal ---------")
    app.run(debug=False,port=80)
