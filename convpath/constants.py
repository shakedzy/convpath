__package_name__ = __name__.split('.')[0]


def _get_version_from_setuptools() -> str:
    from pkg_resources import get_distribution
    global __package_name__
    return get_distribution(__package_name__).version

__version__ = _get_version_from_setuptools()

USER = 'USER'
ASSISTANT = 'ASSISTANT'