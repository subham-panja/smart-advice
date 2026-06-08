from routes.analysis import analysis_bp
from routes.config_routes import config_bp
from routes.orchestrator import orchestrator_bp
from routes.positions import positions_bp


def register_blueprints(app):
    app.register_blueprint(analysis_bp)
    app.register_blueprint(orchestrator_bp)
    app.register_blueprint(positions_bp)
    app.register_blueprint(config_bp)
