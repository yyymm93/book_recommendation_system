from __future__ import annotations

import atexit

from flask import Flask, render_template
from flask_cors import CORS

from config import Settings
from routes.auth_routes import auth_bp
from routes.book_routes import book_bp
from routes.graph_routes import graph_bp
from routes.rating_routes import rating_bp
from routes.recommend_routes import recommend_bp
from services.local_store import init_local_db
from services.model_service import ModelService
from services.neo4j_service import Neo4jService
from services.recommender_service import RecommenderService


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app, resources={r"/*": {"origins": "*"}})

    init_local_db()

    neo = Neo4jService()
    atexit.register(neo.close)
    model = ModelService()
    model.load()
    recommender = RecommenderService(neo4j_service=neo, model_service=model)

    app.extensions["neo4j_service"] = neo
    app.extensions["model_service"] = model
    app.extensions["recommender_service"] = recommender

    app.register_blueprint(auth_bp)
    app.register_blueprint(book_bp)
    app.register_blueprint(rating_bp)
    app.register_blueprint(recommend_bp)
    app.register_blueprint(graph_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return {"code": 0, "message": "ok"}

    @app.teardown_appcontext
    def close_neo4j(_error=None):
        # Driver stays valid for process lifetime; no-op here.
        return None

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=Settings.FLASK_HOST,
        port=Settings.FLASK_PORT,
        debug=Settings.FLASK_DEBUG,
    )
