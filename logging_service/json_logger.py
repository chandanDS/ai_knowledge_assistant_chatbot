# ============================================================
# JSON LOGGER
# ============================================================
#
# Centralized JSON-based interaction logger for the
# AI Knowledge Assistant.
#
# ------------------------------------------------------------
# Responsibilities
# ------------------------------------------------------------
#
# 1. Create the log directory automatically
# 2. Create the JSON log file automatically
# 3. Store one record per chatbot interaction
# 4. Store user and session information
# 5. Store routing information
# 6. Store token consumption
# 7. Store latency
# 8. Provide functions for reading logs
# 9. Provide function for clearing logs
#
# ------------------------------------------------------------
# Log location
# ------------------------------------------------------------
#
#     data/chatbot_logs.json
#
# ------------------------------------------------------------
# Example record
# ------------------------------------------------------------
#
# {
#     "timestamp": "2026-08-14T22:15:10",
#     "session_id": "abc-123",
#     "user_id": "chandan",
#     "query": "Who won the 2023 World Cup?",
#     "route": "WEB_SEARCH",
#     "model": "gpt-4o",
#     "response": "...",
#     "documents_retrieved": 0,
#     "router_input_tokens": 120,
#     "router_output_tokens": 15,
#     "final_input_tokens": 950,
#     "final_output_tokens": 180,
#     "total_tokens": 1265,
#     "latency_seconds": 2.41
# }
#
# ============================================================


# ============================================================
# STANDARD LIBRARY
# ============================================================

import json

import os

import tempfile

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, List, Optional


# ============================================================
# LOG FILE CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Find project root
# ------------------------------------------------------------
#
# json_logger.py is located at:
#
#     project/
#         logging_service/
#             json_logger.py
#
# Therefore:
#
#     Path(__file__).resolve().parent.parent
#
# points to the project root.
#
# ------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ------------------------------------------------------------
# Data directory
# ------------------------------------------------------------

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)


# ------------------------------------------------------------
# JSON log file
# ------------------------------------------------------------

LOG_FILE = (
    DATA_DIR
    / "chatbot_logs.json"
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _ensure_log_file():
    """
    Make sure the data directory and JSON log file exist.
    """

    # --------------------------------------------------------
    # Create data directory
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Create empty JSON array if file doesn't exist
    # --------------------------------------------------------

    if not LOG_FILE.exists():

        with open(
            LOG_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                [],
                file,
                indent=4
            )


# ============================================================
# READ EXISTING LOGS
# ============================================================

def _read_logs_internal() -> List[Dict[str, Any]]:
    """
    Internal function to read all log records.

    Returns
    -------
    list
        List of interaction dictionaries.
    """

    _ensure_log_file()


    try:

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        # ----------------------------------------------------
        # Ensure expected structure
        # ----------------------------------------------------

        if isinstance(
            data,
            list
        ):

            return data


        return []


    except (
        json.JSONDecodeError,
        OSError
    ):

        # ----------------------------------------------------
        # If log file is corrupted or unavailable,
        # don't crash the chatbot.
        # ----------------------------------------------------

        return []


# ============================================================
# WRITE LOGS SAFELY
# ============================================================

def _write_logs_internal(
    logs: List[Dict[str, Any]]
):
    """
    Write the complete log collection to disk.

    Uses a temporary file followed by replacement so that
    the application is less likely to leave a partially
    written JSON file if the process is interrupted.
    """

    _ensure_log_file()


    # --------------------------------------------------------
    # Create temporary file in same directory
    # --------------------------------------------------------

    fd, temp_path = tempfile.mkstemp(

        prefix="chatbot_logs_",

        suffix=".tmp",

        dir=str(DATA_DIR)

    )


    try:

        # ----------------------------------------------------
        # Write JSON
        # ----------------------------------------------------

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(

                logs,

                file,

                indent=4,

                ensure_ascii=False

            )


        # ----------------------------------------------------
        # Replace original file
        # ----------------------------------------------------

        os.replace(

            temp_path,

            LOG_FILE

        )


    except Exception:

        # ----------------------------------------------------
        # Clean up temporary file
        # ----------------------------------------------------

        try:

            os.remove(
                temp_path
            )

        except OSError:

            pass


        raise


# ============================================================
# LOG INTERACTION
# ============================================================

def log_interaction_json(

    session_id: str,

    user_id: str,

    query: str,

    route: str,

    model: str,

    response: str,

    documents_retrieved: int = 0,

    router_input_tokens: int = 0,

    router_output_tokens: int = 0,

    final_input_tokens: int = 0,

    final_output_tokens: int = 0,

    total_tokens: int = 0,

    latency_seconds: float = 0.0,

    **additional_fields

) -> bool:
    """
    Store one chatbot interaction in the JSON log.

    Parameters
    ----------
    session_id:
        Unique conversation/session UUID.

    user_id:
        Logged-in user.

    query:
        User question.

    route:
        Selected route such as:
            RAG_KNOWLEDGE
            WEB_SEARCH
            GENERAL_LLM

    model:
        LLM model used.

    response:
        Final chatbot response.

    documents_retrieved:
        Number of RAG documents retrieved.

    router_input_tokens:
        Router LLM input tokens.

    router_output_tokens:
        Router LLM output tokens.

    final_input_tokens:
        Final LLM input tokens.

    final_output_tokens:
        Final LLM output tokens.

    total_tokens:
        Total tokens consumed.

    latency_seconds:
        End-to-end response latency.

    additional_fields:
        Optional future fields.

    Returns
    -------
    bool
        True if successfully logged.
        False if logging failed.
    """

    try:

        # ====================================================
        # ENSURE FILE EXISTS
        # ====================================================

        _ensure_log_file()


        # ====================================================
        # READ EXISTING LOGS
        # ====================================================

        logs = _read_logs_internal()


        # ====================================================
        # CREATE LOG RECORD
        # ====================================================

        record = {

            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            "timestamp": (
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            ),


            # ------------------------------------------------
            # User / Session
            # ------------------------------------------------

            "session_id": session_id,

            "user_id": user_id,


            # ------------------------------------------------
            # Query
            # ------------------------------------------------

            "query": query,


            # ------------------------------------------------
            # Routing
            # ------------------------------------------------

            "route": route,


            # ------------------------------------------------
            # Model
            # ------------------------------------------------

            "model": model,


            # ------------------------------------------------
            # Response
            # ------------------------------------------------

            "response": response,


            # ------------------------------------------------
            # RAG
            # ------------------------------------------------

            "documents_retrieved": (
                documents_retrieved
            ),


            # ------------------------------------------------
            # Router token usage
            # ------------------------------------------------

            "router_input_tokens": (
                router_input_tokens
            ),

            "router_output_tokens": (
                router_output_tokens
            ),


            # ------------------------------------------------
            # Final LLM token usage
            # ------------------------------------------------

            "final_input_tokens": (
                final_input_tokens
            ),

            "final_output_tokens": (
                final_output_tokens
            ),


            # ------------------------------------------------
            # Total token usage
            # ------------------------------------------------

            "total_tokens": (
                total_tokens
            ),


            # ------------------------------------------------
            # Latency
            # ------------------------------------------------

            "latency_seconds": (
                round(
                    float(
                        latency_seconds
                    ),
                    3
                )
            )
        }


        # ====================================================
        # ADD OPTIONAL FUTURE FIELDS
        # ====================================================

        if additional_fields:

            record.update(
                additional_fields
            )


        # ====================================================
        # APPEND RECORD
        # ====================================================

        logs.append(
            record
        )


        # ====================================================
        # WRITE FILE
        # ====================================================

        _write_logs_internal(
            logs
        )


        return True


    except Exception as exc:

        # ====================================================
        # IMPORTANT
        # ====================================================
        #
        # Logging must NEVER break the chatbot.
        #
        # If logging fails, print the error and allow the
        # chatbot to continue.
        #
        # ====================================================

        print(
            "[LOGGER ERROR] "
            f"{exc}"
        )

        return False


# ============================================================
# PUBLIC READ FUNCTION
# ============================================================

def read_logs(
    session_id: Optional[str] = None,

    user_id: Optional[str] = None,

    route: Optional[str] = None

) -> List[Dict[str, Any]]:
    """
    Read chatbot logs.

    Optional filters:

        session_id
        user_id
        route

    Examples
    --------

    read_logs()

    read_logs(
        session_id="abc-123"
    )

    read_logs(
        user_id="chandan"
    )

    read_logs(
        route="WEB_SEARCH"
    )
    """

    logs = _read_logs_internal()


    # ========================================================
    # SESSION FILTER
    # ========================================================

    if session_id:

        logs = [

            log

            for log in logs

            if log.get(
                "session_id"
            ) == session_id

        ]


    # ========================================================
    # USER FILTER
    # ========================================================

    if user_id:

        logs = [

            log

            for log in logs

            if log.get(
                "user_id"
            ) == user_id

        ]


    # ========================================================
    # ROUTE FILTER
    # ========================================================

    if route:

        logs = [

            log

            for log in logs

            if log.get(
                "route"
            ) == route

        ]


    return logs


# ============================================================
# CLEAR LOGS
# ============================================================

def clear_logs() -> bool:
    """
    Delete all chatbot interaction logs.

    Returns
    -------
    bool
        True if successful.
    """

    try:

        _ensure_log_file()


        _write_logs_internal(
            []
        )


        return True


    except Exception as exc:

        print(
            "[LOGGER ERROR] "
            f"Unable to clear logs: {exc}"
        )

        return False


# ============================================================
# LOG FILE LOCATION
# ============================================================

def get_log_file_path() -> str:
    """
    Return the absolute path of the JSON log file.
    """

    _ensure_log_file()


    return str(
        LOG_FILE
    )