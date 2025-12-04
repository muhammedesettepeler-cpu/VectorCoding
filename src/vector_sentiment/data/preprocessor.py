"""Text preprocessing utilities.

This module provides text preprocessing functions including lowercase conversion,
stopword removal, punctuation removal, and text normalization.
"""

from loguru import logger

from vector_sentiment.config.constants import (
    MULTIPLE_SPACES_PATTERN,
    PUNCTUATION_PATTERN,
    STOPWORDS,
)


class TextPreprocessor:
    """Text preprocessing pipeline.

    This class provides configurable text preprocessing with support for
    lowercase conversion, stopword removal, punctuation removal, and
    whitespace normalization.

    Attributes:
        lowercase: Whether to convert text to lowercase
        remove_stopwords: Whether to remove stopwords
        remove_punctuation: Whether to remove punctuation
        custom_stopwords: Optional custom stopword set

    Example:
        >>> preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
        >>> clean = preprocessor.preprocess("Hello, World!")
        >>> print(clean)  # "hello world"
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        custom_stopwords: set[str] | None = None,
    ) -> None:
        """Initialize text preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_stopwords: Remove stopwords
            remove_punctuation: Remove punctuation
            custom_stopwords: Optional custom stopword set (adds to default)
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation

        # Combine default and custom stopwords
        self.stopwords = STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        logger.info(
            f"Initialized TextPreprocessor: lowercase={lowercase}, "
            f"remove_stopwords={remove_stopwords}, "
            f"remove_punctuation={remove_punctuation}, "
            f"stopwords_count={len(self.stopwords)}"
        )

    def _lowercase(self, text: str) -> str:
        """Convert text to lowercase.

        Args:
            text: Input text

        Returns:
            Lowercase text
        """
        return text.lower()

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text.

        Args:
            text: Input text

        Returns:
            Text without punctuation
        """
        return PUNCTUATION_PATTERN.sub(" ", text)

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text.

        Args:
            text: Input text (should be lowercase if checking lowercase stopwords)

        Returns:
            Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return " ".join(filtered_words)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace by replacing multiple spaces with single space.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        return MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()

    def preprocess(self, text: str) -> str:
        """Preprocess text using configured pipeline.

        Applies preprocessing steps in the following order:
        1. Lowercase conversion (if enabled)
        2. Punctuation removal (if enabled)
        3. Stopword removal (if enabled)
        4. Whitespace normalization (always applied)

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess("This is a test!")
            'test'
        """
        if not text or not text.strip():
            return ""

        processed = text

        # Apply transformations in order
        if self.lowercase:
            processed = self._lowercase(processed)

        if self.remove_punctuation:
            processed = self._remove_punctuation(processed)

        if self.remove_stopwords:
            processed = self._remove_stopwords(processed)

        # Always normalize whitespace
        processed = self._normalize_whitespace(processed)

        return processed

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """Preprocess a batch of texts.

        Args:
            texts: List of texts to preprocess

        Returns:
            List of preprocessed texts

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> batch = ["Hello, world!", "This is a test."]
            >>> preprocessor.preprocess_batch(batch)
            ['hello world', 'test']
        """
        return [self.preprocess(text) for text in texts]
