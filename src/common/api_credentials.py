 # **********************************************************
 #
 # @Author: Andreas Paepcke
 # @Date:   2026-09-08 19:21:17
 # @File:   /Users/paepcke/VSCodeWorkspaces/image-gen-test/src/common/api_credentials.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2026-09-08 19:32:09
 #
 # **********************************************************
"""
Loads API credentials from files under $HOME/.ssh, per convention:
API keys are never put directly in code or passed via environment
variables set ad hoc -- they live in a file under $HOME/.ssh, one
credential per line.

Lives at <proj-root>/src/common/api_credentials.py.
"""

from pathlib import Path


class FileBasedApiCredentials:
    """Reads an API key, and an optional second line, from a local file.

    :param key_path: Path to the credentials file. First line is the
        API key; an optional second line is a secondary value (e.g.
        an organization id).
    :raises FileNotFoundError: if key_path does not exist.
    :raises ValueError: if the file is empty.
    """

    def __init__(self, key_path: Path):
        self.key_path = Path(key_path)
        lines = self.key_path.read_text().splitlines()
        if not lines or not lines[0].strip():
            raise ValueError(f"No API key found in {self.key_path}")
        self.api_key = lines[0].strip()
        self.secondary = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None


class OpenAICredentials(FileBasedApiCredentials):
    """OpenAI credentials: key on line 1, organization id on line 2.

    :param key_path: Defaults to $HOME/.ssh/openai_api_key.txt.
    """

    DEFAULT_PATH = Path.home() / ".ssh" / "openai_api_key.txt"

    def __init__(self, key_path: Path = None):
        super().__init__(key_path or self.DEFAULT_PATH)

    @property
    def organization(self) -> str:
        """The organization id from line 2 of the credentials file.

        :return: Organization id, or None if the file has only one line.
        """
        return self.secondary
