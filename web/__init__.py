from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "221321320@Nc"

    from .views import views

    app.register_blueprint(views, url_prefix="/")

    return app
