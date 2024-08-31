import pkg_resources
from typing import TypeVar, Any
from . import __package_name__


T = TypeVar('T')

def path_to_resource(filename: str) -> str:
    return pkg_resources.resource_filename(__package_name__, f'resources/{filename}')


def flatten(lst: list[list[T]]) -> list[T]:
    return [x for xs in lst for x in xs]


def deflatten(flat_list: list[T], original_list: list[list[Any]]) -> list[list[T]]:
    de_flattened = []
    index = 0

    # Calculate sublist lengths from the original list
    sublist_lengths = [len(sublist) for sublist in original_list]

    for length in sublist_lengths:
        de_flattened.append(flat_list[index:index + length])
        index += length

    return de_flattened
