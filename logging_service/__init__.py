# ============================================================
# LOGGING SERVICE PACKAGE
# ============================================================
#
# Centralized application logging service.
#
# The package currently uses JSON file based logging.
#
# Future implementations can replace this with:
#
#   - PostgreSQL
#   - Oracle
#   - Azure Application Insights
#   - AWS CloudWatch
#   - Elasticsearch
#   - Databricks
#
# without changing the chatbot UI layer.
#
# ============================================================


from .json_logger import (
    log_interaction_json,
    read_logs,
    clear_logs
)


__all__ = [

    "log_interaction_json",

    "read_logs",

    "clear_logs"

]