from dynaconf import Dynaconf
from .singleton import SingletonMeta
from .utils import path_to_resource


class Settings(metaclass=SingletonMeta):
    """
    A Singleton class for settings
    """    
    _settings = None

    def __new__(cls, *args, **kwargs):
        cls._settings = Dynaconf(
            environments=False,
            settings_files=[path_to_resource('default_settings.toml')],
            envvar_prefix='CONVPATH'
        )
        instance = super().__new__(cls, *args, **kwargs)
        return instance
    
    def __getattr__(self, name):
        try:
            return getattr(self._settings, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name == '_settings':
            # Handle setting the '_settings' attribute directly.
            # Use super().__setattr__() to avoid infinite recursion.
            super().__setattr__(name, value)
        else:
            # Attempt to set the attribute on the '_settings' object.
            try:
                setattr(self._settings, name, value)
            except AttributeError:
                # If it fails, set the attribute normally.
                super().__setattr__(name, value)