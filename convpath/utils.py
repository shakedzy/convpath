import pkg_resources
from pydantic import BaseModel
from typing import TypeVar, Any
from .constants import __package_name__


T = TypeVar('T')


def path_to_resource(filename: str) -> str:
    """
    Returns the absolute path to a resource file for the package resources directory

    Parameters
    ----------
    filename : str
        The filename of the resource to be retrieved

    Returns
     -------
        str
            The absolute path to the resource file
    """
    return pkg_resources.resource_filename(__package_name__, f'__resources__/{filename}')


def flatten(lst: list[list[T]]) -> list[T]:
    """
    Flattens a nested list of lists into one

    Parameters
     ----------
        lst : list[list[T]
            The nested lists to be flattened
    
    Returns
    -------
        list[T]
            The flattened list
    """
    return [x for xs in lst for x in xs]


def deflatten(flat_list: list[T], original_shape_list: list[list[Any]]) -> list[list[T]]:
    """
    De-flattens a flattened list of lists into its original nested form

    Parameters
    ----------
    flat_list : list[T]
        The flattened list to be de-flattened
    original_shape_list : list[list[Any]]
        Any list of lists of the original shape that the flattened list was derived from

    Returns
     -------
    list[list[T]]
        The de-flattened nested lists
    """
    de_flattened = []
    index = 0

    # Calculate sublist lengths from the original list
    sublist_lengths = [len(sublist) for sublist in original_shape_list]

    for length in sublist_lengths:
        de_flattened.append(flat_list[index:index + length])
        index += length

    return de_flattened
