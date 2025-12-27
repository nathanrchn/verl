# Copyright 2025 Allen Institute for AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all instructions."""

from .instructions import *


INSTRUCTION_DICT = {
    "count:word_count_range": WordCountRangeChecker,
    "count:unique_word_count" : UniqueWordCountChecker,
    "ratio:stop_words" : StopWordPercentageChecker,
    "ratio:sentence_type" : SentTypeRatioChecker,
    "ratio:sentence_balance" : SentBalanceChecker,
    "count:conjunctions" : ConjunctionCountChecker,
    "count:person_names" : PersonNameCountChecker,
    "ratio:overlap" : NGramOverlapChecker,
    "count:numbers" : NumbersCountChecker,
    "words:alphabet" : AlphabetLoopChecker,
    "words:vowel" : SingleVowelParagraphChecker,
    "words:consonants" : ConsonantClusterChecker,
    "sentence:alliteration_increment" : IncrementingAlliterationChecker,
    "words:palindrome" : PalindromeChecker,
    "count:punctuation" : PunctuationCoverChecker,
    "format:parentheses" : NestedParenthesesChecker,
    "format:quotes" : NestedQuotesChecker,
    "words:prime_lengths" : PrimeLengthsChecker,
    "format:options" : OptionsResponseChecker,
    "format:newline" : NewLineWordsChecker,
    "format:emoji" : EmojiSentenceChecker,
    "ratio:sentence_words" : CharacterCountUniqueWordsChecker,
    "count:words_japanese" : NthWordJapaneseChecker,
    "words:start_verb" : StartWithVerbChecker,
    "words:repeats" : LimitedWordRepeatChecker,
    "sentence:keyword" : IncludeKeywordChecker,
    "count:pronouns" : PronounCountChecker,
    "words:odd_even_syllables" : AlternateParitySyllablesChecker,
    "words:last_first" : LastWordFirstNextChecker,
    "words:paragraph_last_first" : ParagraphLastFirstWordMatchChecker,
    "sentence:increment" : IncrementingWordCountChecker,
    "words:no_consecutive" : NoConsecutiveFirstLetterChecker,
    "format:line_indent" : IndentStairsChecker,
    "format:quote_unquote" : QuoteExplanationChecker,
    "format:list" : SpecialBulletPointsChecker,
    "format:thesis" : ItalicsThesisChecker,
    "format:sub-bullets" : SubBulletPointsChecker,
    "format:no_bullets_bullets" : SomeBulletPointsChecker,
    "custom:multiples" : PrintMultiplesChecker,
    "custom:mcq_count_length": MultipleChoiceQuestionsChecker,
    "custom:reverse_newline": ReverseNewlineChecker,
    "custom:word_reverse": WordReverseOrderChecker,
    "custom:character_reverse": CharacterReverseOrderChecker,
    "custom:sentence_alphabet": SentenceAlphabetChecker,
    "custom:european_capitals_sort": EuropeanCapitalsSortChecker,
    "custom:csv_city": CityCSVChecker,
    "custom:csv_special_character": SpecialCharacterCSVChecker,
    "custom:csv_quotes": QuotesCSVChecker,
    "custom:date_format_list": DateFormatListChecker,
    "count:keywords_multiple" : KeywordsMultipleChecker,
    "words:keywords_specific_position" : KeywordSpecificPositionChecker,
    "words:words_position" : WordsPositionChecker,
    "repeat:repeat_change" : RepeatChangeChecker,
    "repeat:repeat_simple" : RepeatSimpleChecker,
    "repeat:repeat_span" : RepeatSpanChecker,
    "format:title_case" : TitleCaseChecker,
    "format:output_template" : OutputTemplateChecker,
    "format:no_whitespace" : NoWhitespaceChecker,
}
