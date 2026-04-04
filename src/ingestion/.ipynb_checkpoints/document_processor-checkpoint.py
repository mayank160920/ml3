"""
-------------------------------------
Thin orchestration layer that delegates to InputHandler.

DocumentProcessor is the single entry point for loading any supported
document format. It returns a LoadedInput regardless of whether the
source is a PDF or an image file.
"""
from __future__ import annotations

from pathlib import Path

from input.input_handler import InputHandler
from shared_types import LoadedInput


class DocumentProcessor:
    """
    Loads a document (PDF or image) and returns a LoadedInput object.

    This class exists as a clean façade so that pipeline layers above
    only need to call processor.process(path) without caring about the
    input type routing logic.

    Parameters
    ----------
    input_handler : pre-configured InputHandler instance (optional)
    """

    def __init__(self, input_handler: InputHandler | None = None) -> None:
        self.input_handler = input_handler or InputHandler()

    def process(self, input_path: str | Path) -> LoadedInput:
        """
        Load a document and return a unified LoadedInput.

        Parameters
        ----------
        input_path : path to a PDF or image file

        Returns
        -------
        LoadedInput

        Raises
        ------
        FileNotFoundError : if the file does not exist
        ValueError        : if the file extension is not supported
        """
        return self.input_handler.load(input_path)
