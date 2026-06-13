"""
user_ontology
=============
Module enabling cybersecurity analysts to plug in custom ontologies for
PROFILE × ATTACK × OPINION simulation runs.

Public API
----------
validate_ontology_triplet(profile_path, attack_path, opinion_path)
    → ValidationReport

load_user_ontology_triplet(profile_path, attack_path, opinion_path)
    → Dict[str, OntologyTree]

CLI entry point: src/backend/ontology/user/cli.py
"""

from src.backend.ontology.user.validator import (
    ValidationReport,
    validate_ontology_triplet,
    load_user_ontology_triplet,
)

__all__ = [
    "ValidationReport",
    "validate_ontology_triplet",
    "load_user_ontology_triplet",
]
