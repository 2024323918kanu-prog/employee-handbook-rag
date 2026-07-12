import re


class TextCleaner:
    """
    Cleans extracted PDF text while preserving its structure.
    """

    def clean(self, text: str) -> str:
        # Remove extra spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Remove more than two consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing spaces around lines
        text = "\n".join(line.strip() for line in text.splitlines())

        return text.strip()