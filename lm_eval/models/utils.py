from __future__ import annotations

import collections
import fnmatch
import itertools
import logging
import time
from functools import wraps
from typing import (
    Any,
    Iterator,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)


eval_logger = logging.getLogger(__name__)
T = TypeVar("T")


def chunks(iter, n: int = 0, fn=None):
    """
    Divides an iterable into chunks of specified size or based on a given function.
    Useful for batching

    Parameters:
    - iter: The input iterable to be divided into chunks.
    - n: An integer representing the size of each chunk. Default is 0.
    - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

    Returns:
    An iterator that yields chunks of the input iterable.

    Example usage:
    ```
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for chunk in chunks(data, 3):
        print(chunk)
    ```
    Output:
    ```
    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    [10]
    ```
    """
    arr = []
    for i, x in enumerate(iter):
        arr.append(x)
        if len(arr) == (fn(i, iter) if fn else n):
            yield arr
            arr = []

    if arr:
        yield arr


class MultiChoice:
    def __init__(self, choices) -> None:
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values) -> bool:
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                eval_logger.info("Available tasks to choose:")
                for choice in self.choices:
                    eval_logger.info(f"  - {choice}")
                raise ValueError(f"'{value}' is not in task list")
        return True

    def __iter__(self) -> Iterator:
        yield from self.choices


class Grouper:
    """
    takes an array `arr` and function `fn` and returns a dictionary
    with keys fn(ob) for each ob in `arr` and with values `self.arr[key]` a list of all
    objects in `arr` satisfying `key == fn(ob)`.
    """

    def __init__(self, arr, fn) -> None:
        # self.orig_arr = arr
        self.size = len(arr)
        arr = list(enumerate(arr))

        def group_return_dict(arr, fn):
            res = collections.defaultdict(list)

            for ob in arr:
                res[fn(ob)].append(ob)
            return res

        arr = group_return_dict(arr, lambda x: fn(x[1]))

        # self.arr has format Dict[Tuple[int, <entry from orig. arr>]]
        self.arr = arr
        self._grouped = None

    def get_grouped(self):
        # return the contents but not indices for our grouped dict.
        if self._grouped:
            return self._grouped
        grouped = {}
        for key in self.arr.keys():
            # drop the index from each element of self.arr
            grouped[key] = [y[1] for y in self.arr[key]]
        self._grouped = grouped
        return grouped

    def get_original(self, grouped_dict):
        # take in a grouped dictionary with e.g. results for each key listed
        # in the same order as the instances in `self.arr`, and
        # return the results in the same (single list) order as `self.orig_arr`.
        res = [None] * self.size
        cov = [False] * self.size
        # orig = [None] * self.size

        assert grouped_dict.keys() == self.arr.keys()

        for key in grouped_dict.keys():
            for (ind, _), v in zip(self.arr[key], grouped_dict[key], strict=True):
                res[ind] = v
                cov[ind] = True
                # orig[ind] = _

        assert all(cov)
        # assert orig == self.orig_arr

        return res


def undistribute(iterable):
    """
    Undoes https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute .

    Re-interleaves results that have been split using more_itertools.distribute:
        >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 3, 5]
        >>> list(group_2)
        [2, 4, 6]
        >>> undistribute([group_1, group_2])
        [1, 2, 3, 4, 5, 6]

    Handles non-uniform component lengths:

        >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 4, 7], [2, 5], [3, 6]]
        >>> undistribute(children)
        [1, 2, 3, 4, 5, 6, 7]

    Also handles when some iterables are empty:

        >>> children = distribute(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]
        >>> undistribute(children)
        [1, 2, 3]

    """

    return [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(*[list(x) for x in iterable])
        )
        if x is not None
    ]


def retry_on_specific_exceptions(
    on_exceptions: list[type[Exception]],
    max_retries: int | None = None,
    backoff_time: float = 3.0,
    backoff_multiplier: float = 1.5,
    on_exception_callback: Callable[[Exception, float], Any] | None = None,
):
    """Retry on an LLM Provider's rate limit error with exponential backoff
    For example, to use for OpenAI, do the following:
    ```
    from openai import RateLimitError

    # Recommend specifying max_retries to avoid infinite loops!
    @retry_on_specific_exceptions([RateLimitError], max_retries=3)
    def completion(...):
        # Wrap OpenAI completion function here
        ...
    ```
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sleep_time = backoff_time
            attempt = 0
            while max_retries is None or attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except tuple(on_exceptions) as e:
                    if on_exception_callback is not None:
                        on_exception_callback(e, sleep_time)
                    time.sleep(sleep_time)
                    sleep_time *= backoff_multiplier
                    attempt += 1

        return wrapper

    return decorator


class Collator:
    """
    A class for reordering and batching elements of an array.

    This class allows for sorting an array based on a provided sorting function, grouping elements based on a grouping function, and generating batches from the sorted and grouped data.

    Objects of this class have the group_by attribute which determines the method for grouping
    the data while batching it. Three options include "gen_kwargs", "contexts", or None:
        If group_by == "gen_kwargs" then requests will be grouped by gen_kwargs
        If group_by == "contexts" then requests will be grouped by context + cont[:-1]
        If None then requests will just be reordered by length descending.
    """

    def __init__(
        self,
        arr: Sequence[T],
        sort_fn: Callable[[T], Any] = lambda x: x,
        group_fn: Callable[[T], Any] = lambda x: x[1],
        group_by: Literal["gen_kwargs", "contexts"] | None = None,
    ) -> None:
        self._group_by = group_by
        # 0 indices are enumerated indices. Apply functions to original arr.
        self._sort_fn = lambda x: sort_fn(x[1])
        self._group_fn = lambda x: group_fn(x[1])
        self._reorder_indices: list[int] = []
        self._size = len(arr)
        self._arr_with_indices: Union[dict, tuple] = tuple(
            enumerate(arr)
        )  # [indices, (arr)]
        if self._group_by == "contexts":
            self._group_by_context()
        elif self._group_by == "gen_kwargs":
            self._group_by_index()

    def _group_by_index(self) -> None:
        """Group the elements of a list based on their indices."""
        self._arr_with_indices = self.group(
            self._arr_with_indices, fn=self._group_fn, group_by="gen_kwargs"
        )

    def _group_by_context(self) -> None:
        """Group the array with indices by context."""
        self._arr_with_indices = self.group(
            self._arr_with_indices, fn=self._group_fn, group_by="contexts"
        )

    def get_batched(
        self, n: int = 1, batch_fn: Callable[[int, Iterable[T]], int] | None = None
    ) -> Iterator[T]:
        """
        Generates and yields batches from the reordered array. The method of grouping and batching
        depends on the parameter `group_by`.
        If `group_by` is set to "gen_kwargs", it will batch the
        re-ordered values with same gen_kwargs for each batch.
        If `group_by` is "contexts", it caches the requests by context before batching.
        If `group_by` is neither "gen_kwargs" nor "contexts", it yields the reordered array

        Parameters:
        - n (int): The size of each batch. Defaults to 1.
        - batch_fn ([Callable[[int, Iterable], int]] | None): A function to determine the size of
          each batch. Defaults to None.

        Returns:
        Iterator: An iterator over batches of reordered elements grouped as per the `group_by`
                  attribute.

        Yields:
        List of batched elements according to the `group_by` attribute.
        """
        if self._group_by == "gen_kwargs":
            for (
                _,
                values,
            ) in self._arr_with_indices.items():  # type: ignore
                values = self._reorder(values)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch
        elif self._group_by == "contexts":
            # Get one sample from each key.
            # Select longest continuation per group to ensure sufficient context logits
            values = self._reorder(
                [
                    max(value, key=lambda x: len(x[1][-1]))
                    for value in self._arr_with_indices.values()
                ]
            )
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch
        else:
            values = self._reorder(self._arr_with_indices)  # type: ignore
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def get_cache(
        self,
        req_str: tuple = None,
        cxt_toks: list = None,
        cont_toks: list = None,
        logits: Any = None,
    ) -> Iterator:
        """
        Retrieves cached single-token continuations and their associated arguments, updating indices as necessary.
        """
        if self._group_by == "contexts":
            cache_hit: list = self._arr_with_indices.pop(tuple(cxt_toks + cont_toks[:-1]))
            if (cache_size := len(cache_hit)) == 1:
                self._reorder_indices.extend(x[0] for x in cache_hit)
                yield req_str, cont_toks, logits
            else:
                # If we have matching requests then expand the batch dimension (no-op) and
                # yield each along with its corresponding args.
                multilogits = logits.expand(cache_size, -1, -1).chunk(cache_size)
                indices, req_str, cont_toks = zip(
                    *[(x[0], x[1][0], x[-1][-1]) for x in cache_hit], strict=True
                )
                self._reorder_indices.extend(indices)
                yield from zip(req_str, cont_toks, multilogits, strict=True)
        else:
            yield req_str, cont_toks, logits

    def _reorder(self, arr: Union[list, tuple]) -> Iterator:
        """
        Reorders the elements in the array based on the sorting function.

        Parameters:
        - arr (list | tuple): The array or iterable to be reordered.

        Yields:
            Iterator
        """
        arr = sorted(arr, key=self._sort_fn)
        if self._group_by != "contexts":
            # If grouped by contexts then indices will be set in get_cache()
            self._reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def get_original(self, newarr: list) -> list:
        """
        Restores the original order of elements from the reordered list.

        Parameters:
        - newarr (list): The reordered array.

        Returns:
        list: The array with elements restored to their original order.
        """
        res = [None] * self._size
        cov = [False] * self._size

        for ind, v in zip(self._reorder_indices, newarr, strict=True):
            res[ind] = v
            cov[ind] = True

        assert all(cov)

        return res

    def __len__(self):
        return self._size

    @staticmethod
    def group(
        arr: Iterable[T],
        fn: Callable[[T], Sequence[T] | dict],
        group_by: Literal["gen_kwargs", "contexts"] = "gen_kwargs",
    ) -> dict:
        """
        Groups elements of an iterable based on a provided function.


        The `group_by` parameter determines the method of grouping.
        If `group_by` is "contexts", the elements are grouped by [context + cont][:-1].
        If `group_by` is "gen_kwargs", the elements are grouped based on the gen_kwargs dict.

        Parameters:
        - arr (Iterable): The iterable to be grouped.
        - fn (Callable): The function to determine the grouping.
        - values (bool): If True, returns the values of the group. Defaults to False.

        Returns:
        Iterator: An iterable of grouped elements.
        """
        res = collections.defaultdict(list)
        for ob in arr:
            # where ob == [context + cont]
            if group_by == "contexts":
                res[tuple(fn(ob))].append(ob)
            else:
                try:
                    hashable_dict = tuple(
                        (
                            key,
                            tuple(value)
                            if isinstance(value, collections.abc.Iterable)
                            else value,
                        )
                        for key, value in sorted(fn(ob).items())
                    )
                    res[hashable_dict].append(ob)
                except (TypeError, AttributeError):
                    res[tuple(fn(ob))].append(ob)
        return res

    @staticmethod
    def get_chunks(
        _iter, n: int = 0, fn: Callable[[int, Iterable[T]], int] | None = None
    ) -> Iterator[T]:
        """
        Divides an iterable into chunks of specified size or based on a given function.
        Useful for batching

        Parameters:
        - iter: The input iterable to be divided into chunks.
        - n: An integer representing the size of each chunk. Default is 0.
        - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

        Returns:
        An iterator that yields chunks of the input iterable.

        Example usage:
        ```
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for chunk in chunks(data, 3):
            print(chunk)
        ```
        Output:
        ```
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
        ```
        """
        arr = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            arr.append(x)
            if len(arr) == (fn(i, _iter) if fn else n):
                yield arr
                arr = []

        if arr:
            yield arr


def configure_pad_token(tokenizer):
    """
    This function checks if the tokenizer has a padding token and sets it if not present.
    Some tokenizers require special handling.

    Args:
        tokenizer: The tokenizer for which the padding token is to be handled.

    Returns:
        The tokenizer after the padding token has been handled.
    """
    if tokenizer.pad_token:
        pass
    elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # For tokenizers that support adding special tokens
        if hasattr(tokenizer, 'add_special_tokens'):
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    return tokenizer


def handle_stop_sequences(
    until: Union[str, list, None], eos: Optional[str]
) -> list:
    """Ensures that the `until` parameter is a list of stop sequences and includes the EOS token."""
    if isinstance(until, str):
        until = [until]
    elif until is None:
        until = []
    elif not isinstance(until, list):
        raise ValueError(
            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
        )

    if eos is not None and eos not in until:
        until.append(eos)
    return until
