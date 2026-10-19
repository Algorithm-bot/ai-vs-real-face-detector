"""Name normalization for source folders.

Provides a function to normalize source folder names between slash-separated
(progan/airplane) and underscore-separated (progan_airplane) forms.
"""

def normalize_source(name: str) -> str:
    """Normalize a source name to underscore-separated form.

    Args:
        name: Source name, either slash-separated or underscore-separated.

    Returns:
        Underscore-separated form (e.g., "progan/airplane" -> "progan_airplane").
    """
    return name.replace("/", "_")